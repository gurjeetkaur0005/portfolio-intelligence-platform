from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd

from config.asset_classes import ASSET_CLASSES
from config.risk_categories import RISK_CATEGORIES
from src.data.client_profile_generator import generate_client_profiles
from src.data.portfolio_generator import generate_portfolios
from src.database.models import PortfolioHoldingModel
from src.database.repositories import PortfolioRepository
from src.database.session import (
    DatabaseSessionFactory,
    get_database_session_factory,
)
from src.utils.logger import get_logger


SEED_PORTFOLIO_COUNT = 10
SEED_RANDOM_SEED = 20_260_806
SEED_CLIENT_ID_PREFIX = "DEV-C"
SEED_PORTFOLIO_ID_PREFIX = "DEV-P"
DEFAULT_CURRENCY = "USD"
MONEY_QUANTUM = Decimal("0.01")
WEIGHT_QUANTUM = Decimal("0.0000000001")

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Represent the outcome of a development seed run."""

    created: int
    skipped: int
    failed: int


def seed_development_portfolios(
    session_factory: DatabaseSessionFactory | None = None,
) -> SeedResult:
    """
    Seed deterministic development clients, portfolios, and holdings.

    Existing seeded portfolio IDs are skipped so repeated runs are
    idempotent. All writes are committed in one transaction.
    """

    resolved_session_factory = (
        session_factory
        if session_factory is not None
        else get_database_session_factory()
    )

    with resolved_session_factory() as session:
        repository = PortfolioRepository(session)

        try:
            with session.begin():
                result = _seed_with_repository(repository)
        except Exception:
            session.rollback()
            logger.exception("development_seed_failed")
            return SeedResult(
                created=0,
                skipped=0,
                failed=SEED_PORTFOLIO_COUNT,
            )

    logger.info(
        "development_seed_complete created=%s skipped=%s failed=%s",
        result.created,
        result.skipped,
        result.failed,
    )
    return result


def _seed_with_repository(
    repository: PortfolioRepository,
) -> SeedResult:
    """Stage deterministic seed records using the repository layer."""

    client_profiles = _build_seed_client_profiles()
    portfolios = generate_portfolios(
        client_profiles=client_profiles,
        seed=SEED_RANDOM_SEED,
    )

    created = 0
    skipped = 0

    for portfolio_row in portfolios.itertuples(index=False):
        portfolio_id = str(portfolio_row.portfolio_id)

        if (
            repository.get_portfolio_by_business_id(
                portfolio_id
            )
            is not None
        ):
            skipped += 1
            logger.info(
                "development_seed_portfolio_skipped "
                "portfolio_id=%s",
                portfolio_id,
            )
            continue

        client_row = _client_row_for_portfolio(
            client_profiles=client_profiles,
            portfolio_id=portfolio_id,
        )
        client_id = str(client_row["client_id"])
        risk_category = str(client_row["risk_category"])

        client = repository.get_client_by_business_id(
            client_id
        )
        if client is None:
            client = repository.stage_client(
                client_id=client_id,
                risk_category=risk_category,
            )

        portfolio_value = _portfolio_value_for_index(
            created + skipped
        )
        portfolio = repository.stage_portfolio(
            client=client,
            portfolio_id=portfolio_id,
            portfolio_value=portfolio_value,
            currency=DEFAULT_CURRENCY,
        )
        repository.stage_replace_holdings(
            portfolio=portfolio,
            holdings=_build_holdings(
                portfolio_row=portfolio_row,
                portfolio_value=portfolio_value,
            ),
        )

        created += 1
        logger.info(
            "development_seed_portfolio_created "
            "portfolio_id=%s client_id=%s risk_category=%s",
            portfolio_id,
            client_id,
            risk_category,
        )

    return SeedResult(
        created=created,
        skipped=skipped,
        failed=0,
    )


def _build_seed_client_profiles() -> pd.DataFrame:
    """Build deterministic clients covering all configured risk categories."""

    client_profiles = generate_client_profiles(
        number_of_clients=SEED_PORTFOLIO_COUNT,
        seed=SEED_RANDOM_SEED,
    ).copy(deep=True)
    risk_categories = list(RISK_CATEGORIES)

    for index in range(SEED_PORTFOLIO_COUNT):
        client_profiles.loc[index, "client_id"] = (
            f"{SEED_CLIENT_ID_PREFIX}{index + 1:05d}"
        )
        client_profiles.loc[index, "portfolio_id"] = (
            f"{SEED_PORTFOLIO_ID_PREFIX}{index + 1:05d}"
        )
        client_profiles.loc[index, "risk_category"] = (
            risk_categories[index % len(risk_categories)]
        )

    return client_profiles


def _client_row_for_portfolio(
    *,
    client_profiles: pd.DataFrame,
    portfolio_id: str,
) -> pd.Series:
    """Return one generated client row for a portfolio ID."""

    matching_rows = client_profiles.loc[
        client_profiles["portfolio_id"] == portfolio_id
    ]

    if matching_rows.empty:
        raise ValueError(
            f"No seed client found for portfolio {portfolio_id!r}."
        )

    return matching_rows.iloc[0]


def _portfolio_value_for_index(
    index: int,
) -> Decimal:
    """Return a deterministic positive portfolio value."""

    return Decimal(500_000 + (index * 125_000)).quantize(
        MONEY_QUANTUM
    )


def _build_holdings(
    *,
    portfolio_row: Any,
    portfolio_value: Decimal,
) -> list[PortfolioHoldingModel]:
    """Build six holding models from generated portfolio weights."""

    weights = _holding_weights(portfolio_row)

    return [
        PortfolioHoldingModel(
            asset=asset,
            current_weight=weight,
            current_value=_money_value(
                portfolio_value * weight
            ),
            cost_basis=_money_value(
                portfolio_value * weight
            ),
        )
        for asset, weight in zip(
            ASSET_CLASSES,
            weights,
        )
    ]


def _holding_weights(
    portfolio_row: Any,
) -> list[Decimal]:
    """Return generated holding weights rounded to database precision."""

    weights = [
        _weight_value(
            getattr(portfolio_row, f"current_{asset}")
        )
        for asset in ASSET_CLASSES
    ]
    first_weights = weights[:-1]
    final_weight = Decimal("1.0000000000") - sum(
        first_weights,
        Decimal("0"),
    )

    return [
        *first_weights,
        final_weight.quantize(
            WEIGHT_QUANTUM,
            rounding=ROUND_HALF_UP,
        ),
    ]


def _weight_value(
    value: object,
) -> Decimal:
    """Convert a generated weight to database precision."""

    return Decimal(str(value)).quantize(
        WEIGHT_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _money_value(
    value: Decimal,
) -> Decimal:
    """Convert a calculated holding value to database precision."""

    return value.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def main() -> None:
    """Run the development seed script."""

    result = seed_development_portfolios()

    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

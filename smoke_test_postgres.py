from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from src.database.repositories import PortfolioRepository
from src.database.session import get_database_session_factory


def main() -> None:
    """Verify that PostgreSQL can save and read portfolio records."""

    session_factory = get_database_session_factory()

    unique_suffix = uuid4().hex[:8].upper()

    client_business_id = f"C-SMOKE-{unique_suffix}"
    portfolio_business_id = f"P-SMOKE-{unique_suffix}"

    with session_factory() as session:
        repository = PortfolioRepository(session)

        client = repository.create_client(
            client_id=client_business_id,
            risk_category="balanced",
        )

        portfolio = repository.create_portfolio(
            client=client,
            portfolio_id=portfolio_business_id,
            portfolio_value=Decimal("1000000.00"),
            currency="USD",
        )

        loaded_portfolio = (
            repository.require_portfolio_by_business_id(
                portfolio_business_id
            )
        )

        print("PostgreSQL smoke test successful")
        print(f"Client database ID: {client.id}")
        print(f"Client business ID: {client.client_id}")
        print(f"Portfolio database ID: {portfolio.id}")
        print(
            f"Portfolio business ID: "
            f"{loaded_portfolio.portfolio_id}"
        )
        print(
            f"Portfolio value: "
            f"{loaded_portfolio.portfolio_value}"
        )
        print(f"Currency: {loaded_portfolio.currency}")
        print(
            f"Linked client ID: "
            f"{loaded_portfolio.client_id}"
        )


if __name__ == "__main__":
    main()
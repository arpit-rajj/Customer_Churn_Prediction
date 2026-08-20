import argparse
import logging
from src.db.connection import engine, Base
from src.db.models import Customer
from sqlalchemy.orm import Session
from src.data.ingestion import load_and_validate_data, prepare_for_db_insertion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest bank customer data into PostgreSQL")
    parser.add_argument("--filepath", type=str, default="data/raw/churn.csv", help="Path to raw CSV file")
    args = parser.parse_args()

    # 1. Create tables if they don't exist
    logger.info("Ensuring database tables exist...")
    Base.metadata.create_all(bind=engine)

    # 2. Load and validate
    df = load_and_validate_data(args.filepath)
    records = prepare_for_db_insertion(df)

    # 3. Insert into database
    logger.info(f"Inserting {len(records)} records into the database...")
    with Session(engine) as session:
        # Check if data already exists to avoid duplicates if run multiple times
        existing_count = session.query(Customer).count()
        if existing_count > 0:
            logger.warning(f"Found {existing_count} existing records in 'customers' table. Truncating table before insertion.")
            session.query(Customer).delete()
            session.commit()
            
        # Bulk insert
        session.bulk_insert_mappings(Customer, records)
        session.commit()
        
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment
# Use pymysql dialect for MySQL connections
db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("mysql://"):
    # Convert mysql:// to mysql+pymysql:// for SQLAlchemy
    DATABASE_URL = db_url.replace("mysql://", "mysql+pymysql://")
else:
    # Fallback to SQLite
    DATABASE_URL = "sqlite:///./ai_agent_sim.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for getting DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

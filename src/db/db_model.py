from sqlalchemy import REAL, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from config.config import settings

# Base class for all the tables in our database
class Base(DeclarativeBase):
    pass

# Child class for our housing database inheriting from Base class
class Housing(Base):
    __tablename__ = settings.housing_tablename

    longitude: Mapped[float] = mapped_column(REAL(), primary_key=True)
    latitude: Mapped[float] = mapped_column(REAL())
    housing_median_age: Mapped[float] = mapped_column(REAL())
    total_rooms: Mapped[float] = mapped_column(REAL())
    total_bedrooms: Mapped[float] = mapped_column(REAL())
    population: Mapped[float] = mapped_column(REAL())
    households: Mapped[float] = mapped_column(REAL())
    median_income: Mapped[float] = mapped_column(REAL())
    median_house_value: Mapped[float] = mapped_column(REAL())
    ocean_proximity: Mapped[str] = mapped_column(VARCHAR())

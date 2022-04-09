from sqlalchemy import Column, Integer, Text, String, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Trajectory(Base):
    __tablename__ = "trajectory"
    id = Column(Integer, primary_key=True)
    length = Column(Integer)
    start_time = Column(Integer)
    end_time = Column(Integer)
    points = Column(Text)
    embedding = Column(Text)

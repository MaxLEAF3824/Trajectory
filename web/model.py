from sqlalchemy import Column, Integer, Text, String, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Trajectory(Base):
    __tablename__ = "trajectory"
    id = Column(Integer, primary_key=True)
    length = Column(Integer)
    discrete_points = Column(Text)
    spherical_points = Column(Text)

    def __repr__(self):
        return f"id(id={self.id!r}, length={self.length!r}, discrete_points={self.discrete_points!r}, " \
               f"spherical_points={self.spherical_points!r}) "


class OriginalTrajectoryPoint(Base):
    __tablename__ = "original_trajectory"
    id = Column(String(32), primary_key=True)
    time = Column(Integer, primary_key=True)
    longitude = Column(Float)
    latitude = Column(Float)

    def __repr__(self):
        return f"id(id={self.id!r}, time={self.time!r}, longitude={self.longitude!r}, latitude={self.latitude!r})"


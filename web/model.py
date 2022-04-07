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

    def __repr__(self):
        return f"id(id={self.id!r}, length={self.length!r}, start_time={self.start_time}, end_time={self.end_time} " \
               f"points={self.points!r}, embedding={self.embedding}) "


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5]
    t = Trajectory(id=1, length=len(a), start_time=a[0], end_time=a[-1], points=a, embedding="")
    print(t.points)
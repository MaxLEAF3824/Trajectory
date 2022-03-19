from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from typing import List
from model import Trajectory, OriginalTrajectoryPoint


class Mapper:
    def __init__(self):
        self.engine = create_engine("mysql+mysqldb://root:admin@localhost:3306/trajectory", echo=True, future=True)

    def insert_trajectorys(self, trajectories: List[Trajectory]):
        session = Session(self.engine)
        session.add_all(trajectories)
        session.commit()
        return 0

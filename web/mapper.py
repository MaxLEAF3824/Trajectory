from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typing import List

from model import Trajectory, OriginalTrajectoryPoint


class Mapper:
    def __init__(self, db_path="yqguo:guoyiqiu@202.117.43.251:3306/trajectory"):
        self.engine = create_engine(f"mysql+mysqldb://{db_path}", echo=True, future=True)

    def insert_trajectories(self, trajectories: List[Trajectory]):
        session = Session(self.engine)
        session.add_all(trajectories)
        session.commit()
        return 0

    def get_trajectories_all(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory).all()

    def get_trajectory_by_id(self, traj_id):
        session = Session(self.engine)
        return session.query(Trajectory).filter(Trajectory.id == traj_id).first()

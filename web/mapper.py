from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, load_only
from typing import List

from model import Trajectory


class Mapper:
    def __init__(self, db_path="yqguo:guoyiqiu@202.117.43.251:3306/trajectory"):
        self.engine = create_engine(f"mysql+mysqldb://{db_path}", echo=True, future=True)

    def insert_trajectories(self, trajectories: List[Trajectory]):
        session = Session(self.engine)
        session.add_all(trajectories)
        session.commit()
        return 0

    def get_all_trajectories_all(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory).all()

    def get_all_trajectories_points(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.points).all()

    def get_trajectories_points_by_time_slice(self, start_time, end_time):
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.points).filter(
            Trajectory.start_time < end_time).filter(start_time < Trajectory.end_time).all()

    def get_trajectories_embedding_by_time_slice(self, start_time, end_time):
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.embedding).filter(
            Trajectory.start_time < end_time).filter(start_time < Trajectory.end_time).all()

    def get_all_trajectories_embedding(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.embedding).all()

    def get_trajectory_by_id(self, traj_id):
        session = Session(self.engine)
        return session.query(Trajectory).filter(Trajectory.id == traj_id).first()

    def get_trajectory_by_id_list(self, id_list):
        session = Session(self.engine)
        unsorted_trajectories = session.query(Trajectory).options(
            load_only(Trajectory.id, Trajectory.length, Trajectory.start_time, Trajectory.end_time,
                      Trajectory.points)).filter(Trajectory.id.in_(id_list)).all()
        trajectories = []
        for tid in id_list:
            for traj in unsorted_trajectories:
                if traj.id == tid:
                    trajectories.append(traj)
                    break
        return trajectories

    def update_trajectory_embedding_by_id(self, tid, embedding):
        session = Session(self.engine)
        session.query(Trajectory).filter(Trajectory.id == tid).update({Trajectory.embedding: embedding})
        session.commit()
        return 0


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
        return session.query(Trajectory).options(
            load_only(Trajectory.id, Trajectory.length, Trajectory.start_time, Trajectory.end_time,
                      Trajectory.points)).filter(Trajectory.id.in_(id_list)).all()

    def update_trajectory_embedding(self, tid, embedding):
        session = Session(self.engine)
        session.query(Trajectory).filter(Trajectory.id == tid).update({Trajectory.embedding: embedding})
        session.commit()
        return 0


if __name__ == "__main__":
    mapper = Mapper()
    all_traj = mapper.get_all_trajectories_points()
    start_time = 1478048727
    end_time = 1478049242
    id_list = [1, 2, 3]
    embeddings = ["123"] * len(id_list)
    for i in id_list:
        mapper.update_trajectory_embedding(i, embeddings[i - 1])
    # traj_time_slice = mapper.get_trajectories_points_by_time_slice(start_time, end_time)
    # traj_id_list = mapper.get_trajectory_by_id_list(id_list)
    # for traj_id, traj in all_traj:
    #     print(traj_id, traj)
    #     break

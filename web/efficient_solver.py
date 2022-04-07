import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


class EfficientSolver:
    def __init__(self, model_path, device='cpu'):
        self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()
        self.t2g = self.model.t2g
        self.mean_x = self.model.mean_x
        self.mean_y = self.model.mean_y
        self.std_x = self.model.std_x
        self.std_y = self.model.std_y
        self.device = device

    def normalize_traj(self, traj):
        """
        归一化轨迹
        :param traj:
        :return:
        """
        return [((i[0] - self.mean_x) / self.std_x, (i[1] - self.mean_y) / self.std_y) for i in traj]

    # TODO: 实现相似度计算函数
    def compute_similarity(self, query_embeddings, embeddings) -> np.array:
        """
        计算轨迹之间的相似度
        :param query_embeddings: Array [bsz1, embedding_size]
        :param embeddings: Array [bsz2, embedding_size]
        :return:similarity matrix: Array [bsz1, bsz2]
        """
        pass

    # TODO: 实现单条轨迹编码函数
    def embed_trajectory(self, trajectory):
        """
        生成单个轨迹的embedding
        :param trajectory: 经纬度坐标轨迹
        :return:embedding: np.array
        """
        traj_1d, coord_traj = self.t2g.convert1d(trajectory)
        traj_1d = torch.tensor(traj_1d, dtype=torch.long).unsqueeze(0)
        normalized_traj = torch.tensor(self.normalize_traj(coord_traj), dtype=torch.float32).unsqueeze(0)
        traj_lens = torch.tensor([len(normalized_traj)], dtype=torch.long)
        embedding = self.model.forward(traj_1d, normalized_traj, traj_lens)
        return embedding.detach().numpy()[0]

    # TODO: 实现批量轨迹编码函数
    def embed_trajectory_batch(self, trajectories):
        """
        批量生成embedding
        :param trajectories: 经纬度坐标轨迹
        :return:embedding: List[np.array]
        """
        traj_1ds = []
        coord_trajs = []
        for trajectory in trajectories:
            traj_1d, coord_traj = self.t2g.convert1d(trajectory)
            traj_1ds.append(traj_1d)
            coord_trajs.append(coord_traj)
        traj_1d = [torch.tensor(traj_1d, dtype=torch.long) for traj_1d in traj_1ds]
        normalized_traj = [torch.tensor(self.normalize_traj(coord_traj), dtype=torch.float32) for coord_traj in coord_trajs]
        traj_lens = torch.tensor([traj.shape[0] for traj in normalized_traj], dtype=torch.long)
        traj_1d = rnn_utils.pad_sequence(traj_1d, batch_first=True, padding_value=-1)
        normalized_traj = rnn_utils.pad_sequence(normalized_traj, batch_first=True, padding_value=0)
        embedding = self.model.forward(traj_1d, normalized_traj, traj_lens)
        return embedding.detach().numpy()

    # TODO: 实现faiss查询的函数
    def query_faiss(self, query_traj, k):
        """
        利用faiss框架加速查询过程
        :param query_traj:
        :param k:
        :return:
        """
        pass


if __name__ == '__main__':
    test_trajs = [[(104.04668, 30.65522), (104.0465, 30.65552), (104.04637, 30.65573), (104.04619, 30.65602),
                  (104.04605, 30.65624), (104.04595, 30.65642), (104.0458, 30.65667), (104.04562, 30.65698),
                  (104.0455, 30.65715), (104.04536, 30.65735), (104.04518, 30.65756), (104.04488, 30.65781),
                  (104.04465, 30.65795), (104.04442, 30.65805), (104.04424, 30.6581)],
                 [(104.04668, 30.65522), (104.0465, 30.65552), (104.04637, 30.65573), (104.04619, 30.65602),
                  (104.04605, 30.65624), (104.04595, 30.65642), (104.0458, 30.65667), (104.04562, 30.65698),
                  (104.0455, 30.65715), (104.04536, 30.65735)]]
    solver = EfficientSolver('../model/archived_model/model_baseline_rank_12.034.pth')
    emb = solver.embed_trajectory_batch(test_trajs)
    print(emb.shape)

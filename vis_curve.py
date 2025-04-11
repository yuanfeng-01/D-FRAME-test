import os
import pickle
import time
import numpy as np
from itertools import combinations


def calculate_angle(vec1, vec2):
    """计算两向量之间的夹角余弦值"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1  # 避免除零，返回夹角余弦最大值（cos 0 = 1）
    cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    return cosine_angle

def connect_edges(points, directions, k=1, distance_threshold=0.05, angle_threshold=0.8, region_consistency_threshold=0.8):
    """
    改进的邻居连接算法，考虑区域一致性以避免跨区域错误连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - directions: numpy.ndarray，形状为 (N, 3)，方向向量。
    - k: int，选择余弦相似度最高的 k 个近邻。
    - distance_threshold: float，近邻点的最大距离。
    - angle_threshold: float，方向向量的余弦相似度阈值。
    - region_consistency_threshold: float，区域一致性检查阈值（方向变化率）。

    返回：
    - edges: list of tuple，边的连接关系 [(i, j), ...]。
    """
    edges = set()  # 用于存储无向边，避免重复
    num_points = points.shape[0]

    for i in range(num_points):
        point = points[i]
        direction = directions[i]
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            continue  # 忽略零方向向量

        # 计算到所有其他点的向量和距离
        vectors_to_neighbors = points - point  # 向量
        distances = np.linalg.norm(vectors_to_neighbors, axis=1)  # 距离

        # 排除自身和超过距离阈值的点
        valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
        valid_vectors = vectors_to_neighbors[valid_indices]

        # 计算方向相似性和局部一致性
        valid_cosines = []
        for j, vector_to_neighbor in zip(valid_indices, valid_vectors):
            vector_norm = np.linalg.norm(vector_to_neighbor)
            if vector_norm == 0:
                continue  # 忽略重叠点

            # 计算余弦相似性
            cosine_similarity = np.dot(direction, vector_to_neighbor) / (direction_norm * vector_norm)

            # 计算区域一致性
            neighbor_direction = directions[j]
            neighbor_norm = np.linalg.norm(neighbor_direction)
            if neighbor_norm == 0:
                continue

            region_consistency = np.dot(direction, neighbor_direction) / (direction_norm * neighbor_norm)

            # 筛选符合条件的邻居点
            if cosine_similarity >= angle_threshold and region_consistency >= region_consistency_threshold:
                valid_cosines.append((j, cosine_similarity))

        # 按余弦相似度排序，选择最相似的 k 个点
        valid_cosines = sorted(valid_cosines, key=lambda x: x[1], reverse=True)[:k]

        for j, _ in valid_cosines:
            # 确保无向边不重复
            edge = tuple(sorted((i, j)))
            if edge not in edges:
                edges.add(edge)

    return list(edges)

def connect_edges_v2(points, directions, k=1, distance_threshold=0.05, angle_threshold=0.8, region_consistency_threshold=0.8):
    """
    改进的邻居连接算法，选取满足余弦相似度约束且符合区域一致性的距离最近的点连接，避免跳跃连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - directions: numpy.ndarray，形状为 (N, 3)，方向向量。
    - k: int，选择余弦相似度最高的 k 个近邻。
    - distance_threshold: float，近邻点的最大距离。
    - angle_threshold: float，方向向量的余弦相似度阈值。
    - region_consistency_threshold: float，区域一致性检查阈值（方向变化率）。

    返回：
    - edges: list of tuple，边的连接关系 [(i, j), ...]。
    """
    edges = set()  # 用于存储无向边，避免重复
    num_points = points.shape[0]

    for i in range(num_points):
        point = points[i]
        direction = directions[i]
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            continue  # 忽略零方向向量

        # 计算到所有其他点的向量和距离
        vectors_to_neighbors = points - point  # 向量
        distances = np.linalg.norm(vectors_to_neighbors, axis=1)  # 距离

        # 排除自身和超过距离阈值的点
        valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
        valid_vectors = vectors_to_neighbors[valid_indices]
        valid_distances = distances[valid_indices]

        # 计算方向相似性和区域一致性
        valid_neighbors = []
        for j, vector_to_neighbor, distance in zip(valid_indices, valid_vectors, valid_distances):
            vector_norm = np.linalg.norm(vector_to_neighbor)
            if vector_norm == 0:
                continue  # 忽略重叠点

            # 计算余弦相似性
            cosine_similarity = np.dot(direction, vector_to_neighbor) / (direction_norm * vector_norm)

            # 计算区域一致性
            neighbor_direction = directions[j]
            neighbor_norm = np.linalg.norm(neighbor_direction)
            if neighbor_norm == 0:
                continue

            region_consistency = np.dot(direction, neighbor_direction) / (direction_norm * neighbor_norm)

            if cosine_similarity >= angle_threshold and region_consistency >= region_consistency_threshold:
                valid_neighbors.append((j, distance, cosine_similarity))

        # 按距离排序，选择最近的点
        valid_neighbors = sorted(valid_neighbors, key=lambda x: x[1])[:k]

        # 防止跳跃连接，按相邻顺序逐一连接
        for idx in range(len(valid_neighbors)):
            current_neighbor = valid_neighbors[idx][0]
            edge = tuple(sorted((i, current_neighbor)))

            if idx > 0:  # 确保当前点和上一个点直接连接
                previous_neighbor = valid_neighbors[idx - 1][0]
                additional_edge = tuple(sorted((previous_neighbor, current_neighbor)))
                edges.add(additional_edge)

            # 添加当前点的边
            edges.add(edge)

    return list(edges)

def connect_edges_v3(points, directions, k=1, distance_threshold=0.05, angle_threshold=0.9, region_consistency_threshold=0.9):
    """
    改进的邻居连接算法，选取满足余弦相似度约束且符合区域一致性的距离最近的点连接，避免跳跃连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - directions: numpy.ndarray，形状为 (N, 3)，方向向量。
    - k: int，选择余弦相似度最高的 k 个近邻。
    - distance_threshold: float，近邻点的最大距离。
    - angle_threshold: float，方向向量的余弦相似度阈值。
    - region_consistency_threshold: float，区域一致性检查阈值（方向变化率）。

    返回：
    - edges: list of tuple，边的连接关系 [(i, j), ...]。
    """
    edges = set()  # 用于存储无向边，避免重复
    num_points = points.shape[0]

    for i in range(num_points):
        point = points[i]
        direction = directions[i]
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            continue  # 忽略零方向向量

        # 计算到所有其他点的向量和距离
        vectors_to_neighbors = points - point  # 向量
        distances = np.linalg.norm(vectors_to_neighbors, axis=1)  # 距离

        # 排除自身和超过距离阈值的点
        valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
        valid_indices = [valid_idx for valid_idx in valid_indices if (i, valid_idx) not in edges and (valid_idx, i) not in edges]
        valid_vectors = vectors_to_neighbors[valid_indices]
        valid_distances = distances[valid_indices]

        # 计算方向相似性和区域一致性
        valid_neighbors = []
        for j, vector_to_neighbor, distance in zip(valid_indices, valid_vectors, valid_distances):
            vector_norm = np.linalg.norm(vector_to_neighbor)
            if vector_norm == 0:
                continue  # 忽略重叠点

            # 计算余弦相似性
            cosine_similarity = np.abs(np.dot(direction, vector_to_neighbor) / (direction_norm * vector_norm))

            # 计算区域一致性
            neighbor_direction = directions[j]
            neighbor_norm = np.linalg.norm(neighbor_direction)
            if neighbor_norm == 0:
                continue

            region_consistency = np.abs(np.dot(direction, neighbor_direction) / (direction_norm * neighbor_norm))

            if cosine_similarity >= angle_threshold and region_consistency >= region_consistency_threshold:
                valid_neighbors.append((j, distance, cosine_similarity))

        # 按距离排序，选择最近的点
        valid_neighbors = sorted(valid_neighbors, key=lambda x: x[1])[:k]

        # 防止跳跃连接，按相邻顺序逐一连接
        for idx in range(len(valid_neighbors)):
            current_neighbor = valid_neighbors[idx][0]
            edge = tuple(sorted((i, current_neighbor)))

            if idx > 0:  # 确保当前点和上一个点直接连接
                previous_neighbor = valid_neighbors[idx - 1][0]
                additional_edge = tuple(sorted((previous_neighbor, current_neighbor)))
                edges.add(additional_edge)

            # 添加当前点的边
            edges.add(edge)

    return list(edges)

def connect_edges_v4(points, directions, k=2, distance_threshold=0.05, angle_threshold=0.9, region_consistency_threshold=0.9):
    """
    改进的邻居连接算法，选取满足余弦相似度约束且符合区域一致性的距离最近的两个未连接点进行连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - directions: numpy.ndarray，形状为 (N, 3)，方向向量。
    - k: int，选择满足条件的最近邻个数（默认2）。
    - distance_threshold: float，近邻点的最大距离。
    - angle_threshold: float，方向向量的余弦相似度阈值。
    - region_consistency_threshold: float，区域一致性检查阈值（方向变化率）。

    返回：
    - edges: list of tuple，边的连接关系 [(i, j), ...]。
    """
    edges = set()  # 存储无向边，避免重复
    num_points = points.shape[0]

    for i in range(num_points):
        point = points[i]
        direction = directions[i]
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            continue  # 忽略零方向向量

        # 计算到所有其他点的向量和距离
        vectors_to_neighbors = points - point  # 向量
        distances = np.linalg.norm(vectors_to_neighbors, axis=1)  # 距离

        # 排除自身和超过距离阈值的点
        valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
        valid_vectors = vectors_to_neighbors[valid_indices]
        valid_distances = distances[valid_indices]

        # 筛选满足余弦相似度和区域一致性约束的邻居
        valid_neighbors = []
        for j, vector_to_neighbor, distance in zip(valid_indices, valid_vectors, valid_distances):
            vector_norm = np.linalg.norm(vector_to_neighbor)
            if vector_norm == 0:
                continue  # 忽略重叠点

            # 计算方向相似性
            cosine_similarity = np.abs(np.dot(direction, vector_to_neighbor) / (direction_norm * vector_norm))

            # 计算区域一致性
            neighbor_direction = directions[j]
            neighbor_norm = np.linalg.norm(neighbor_direction)
            if neighbor_norm == 0:
                continue

            region_consistency = np.abs(np.dot(direction, neighbor_direction) / (direction_norm * neighbor_norm))

            # 判断约束条件
            if cosine_similarity >= angle_threshold and region_consistency >= region_consistency_threshold:
                valid_neighbors.append((j, distance, cosine_similarity))

        # 排序并选取距离最近的两个未连接的邻居
        valid_neighbors = sorted(valid_neighbors, key=lambda x: x[1])

        for neighbor_info in valid_neighbors[:k]:
            j = neighbor_info[0]
            edge = tuple(sorted((i, j)))

            # 如果尚未与该邻居连接
            if edge not in edges:
                edges.add(edge)

    return list(edges)

def remove_outliers(points, edges):
    edges = set(edges)

    for edge in edges:
        if edge[0] > edge[1]:
            edges.discard(edge)

    num_points = points.shape[0]

    # 构建邻接表
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    # 检查度大于等于 2 的点，并优化其连接
    to_delete_pts = []
    for node in range(num_points):
        neighbors = adjacency_list[node]
        if len(neighbors) < 2:
            continue

        # 计算点与其邻居的边向量
        neighbor_vectors = {neighbor: points[neighbor] - points[node] for neighbor in neighbors}
        neighbor_distances = {neighbor: np.linalg.norm(vec) for neighbor, vec in neighbor_vectors.items()}

        to_remove = []
        to_add = []

        for i, neighbor1 in enumerate(neighbors):
            if neighbor1 in to_delete_pts:
                continue
            for neighbor2 in neighbors[i + 1:]:
                vector1 = neighbor_vectors[neighbor1]
                vector2 = neighbor_vectors[neighbor2]
                vector3 = points[neighbor2] - points[neighbor1]

                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)
                norm3 = np.linalg.norm(vector3)
                if norm1 == 0 or norm2 == 0 or norm3 == 0:
                    continue

                # 计算夹角的余弦值
                cosine_angle = np.dot(vector1, vector2) / (norm1 * norm2)

                if cosine_angle > 0.98:  # 夹角过小
                    # 删除较长的边并连接两个邻居
                    if neighbor_distances[neighbor1] > neighbor_distances[neighbor2]:
                        to_remove.append(tuple(sorted((node, neighbor1))))

                        cosine_angle_23 = np.dot(vector2, -vector3) / (norm2 * norm3)
                        if cosine_angle_23 > 0.9:
                            to_add.append(tuple(sorted((neighbor1, neighbor2))))
                        else:
                            to_delete_pts.append(neighbor1)
                            for neighbor3 in adjacency_list[neighbor1]:
                                to_remove.append(tuple(sorted((neighbor1, neighbor3))))
                    else:
                        to_remove.append(tuple(sorted((node, neighbor2))))

                        cosine_angle_13 = np.dot(vector1, vector3) / (norm1 * norm3)
                        if cosine_angle_13 > 0.9:
                            to_add.append(tuple(sorted((neighbor1, neighbor2))))
                        else:
                            to_delete_pts.append(neighbor2)
                            for neighbor3 in adjacency_list[neighbor2]:
                                to_remove.append(tuple(sorted((neighbor2, neighbor3))))

        # 更新邻接矩阵和边集合
        if len(to_remove) > 0:
            for edge in to_remove:
                if edge in edges:
                    edges.discard(edge)
                    adjacency_list[edge[0]].remove(edge[1])
                    adjacency_list[edge[1]].remove(edge[0])

        if len(to_add) > 0:
            for edge in to_add:
                if edge not in edges:
                    edges.add(edge)
                    if edge[1] not in adjacency_list[edge[0]]:
                        adjacency_list[edge[0]].append(edge[1])
                    if edge[0] not in adjacency_list[edge[1]]:
                        adjacency_list[edge[1]].append(edge[0])

    # 从points中删除to_delete_pts中的所有点，并更新边和邻接表
    to_delete_pts = set(to_delete_pts)
    new_points = []
    new_index_mapping = {}
    updated_edges = set()

    # 新点索引映射
    for i in range(num_points):
        if i not in to_delete_pts:
            new_index_mapping[i] = len(new_points)
            new_points.append(points[i])

    # 更新edges和邻接表
    for edge in edges:
        if edge[0] not in to_delete_pts and edge[1] not in to_delete_pts:
            updated_edges.add((new_index_mapping[edge[0]], new_index_mapping[edge[1]]))

    # 返回更新后的points和edges
    points = np.array(new_points)  # 更新points
    return points, list(updated_edges)  # 返回更新后的points和edges

def remove_high_degree_edges(points, edges):
    """
    删除度数大于 2 的边，动态更新节点的度数，确保每次删除后度数变化被及时考虑。

    参数:
    - new_edges: set of tuple，当前的边集合。
    - num_points: int，点的总数。

    返回:
    - updated_edges: set of tuple，删除多余边后的边集合。
    """
    num_points = points.shape[0]
    # 初始化度数字典
    edge_counts = {i: 0 for i in range(num_points)}
    for edge in edges:
        edge_counts[edge[0]] += 1
        edge_counts[edge[1]] += 1

    # 转换边集合为列表以便逐一删除
    edges_to_check = list(edges)
    updated_edges = set(edges)

    # 持续删除满足条件的边并动态更新度数
    for edge in edges_to_check:
        node_a, node_b = edge

        # 如果两个节点的度数都大于 2，则删除该边
        if edge_counts[node_a] > 2 and edge_counts[node_b] > 2:
            updated_edges.discard(edge)  # 删除边

            # 更新度数
            edge_counts[node_a] -= 1
            edge_counts[node_b] -= 1

    return updated_edges

def extend_isolated_points(points, directions, edges, extended_distance_threshold=0.1, extended_angle_threshold=0.8):
    """
    改进的孤立点处理算法，考虑区域一致性和方向变化。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - directions: numpy.ndarray，形状为 (N, 3)，方向向量。
    - edges: list of tuple，初始边的连接关系。
    - extended_distance_threshold: float，扩展时的距离阈值。
    - extended_angle_threshold: float，扩展时的余弦相似度阈值。

    返回：
    - edges: list of tuple，扩展后的边的连接关系。
    """
    edge_dict = {i: [] for i in range(points.shape[0])}
    for i, j in edges:
        edge_dict[i].append(j)
        edge_dict[j].append(i)

    new_edges = set(edges)  # 初始化为已有的边

    num = 0
    for i, neighbors in edge_dict.items():
        if len(neighbors) == 1:  # 找到只连接一个点的孤立点
            point = points[i]
            direction = directions[i]
            direction_norm = np.linalg.norm(direction)

            if direction_norm == 0:
                continue  # 忽略零方向向量

            # 计算到所有其他点的向量和距离
            vectors_to_neighbors = points - point  # 向量
            distances = np.linalg.norm(vectors_to_neighbors, axis=1)  # 距离

            # 排除自身和已有连接点
            valid_indices = np.where((distances > 0) & (distances <= extended_distance_threshold))[0]
            valid_indices = [j for j in valid_indices if j not in neighbors]

            if not valid_indices:
                continue  # 没有符合条件的点

            # 筛选出符合余弦阈值的点
            valid_cosines = [
                (j, distances[j], np.abs(np.dot(direction, points[j] - point)) /
                 (direction_norm * np.linalg.norm(points[j] - point)))
                for j in valid_indices
            ]
            valid_cosines = [(j, d, cos) for j, d, cos in valid_cosines if cos >= extended_angle_threshold]

            if not valid_cosines:
                continue  # 没有符合条件的点

            # 选择符合条件的最近点
            best_neighbor = min(valid_cosines, key=lambda x: x[1])  # 按距离排序
            j = best_neighbor[0]

            # 确保无向边不重复
            edge = tuple(sorted((i, j)))
            if edge not in new_edges:
                new_edges.add(edge)
                num += 1

    print(f'Added {num} edges')
    return list(new_edges)

def extend_isolated_points_v2(points, edges, distance_threshold=0.05):
    """
    对所有在边中只出现一次的点，
    计算与当前边投影距离最短的孤立点并将两者连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - edges: list of tuple，初始边的连接关系。

    返回：
    - extended_edges: list of tuple，扩展后的边的连接关系。
    """
    edges = set(edges)
    num_points = points.shape[0]

    # 构建邻接表
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    # 查找点数小于 5 的孤立边
    def find_isolated_edges():
        visited = set()
        isolated_edges = []

        def dfs(node, path):
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for neighbor in adjacency_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, path)

        for i in range(num_points):
            if i not in visited:
                path = []
                dfs(i, path)
                if 1 < len(path) < 5:
                    isolated_edges.append(path)

        return isolated_edges

    isolated_edge_groups = find_isolated_edges()

    # 重新连接孤立边
    for group in isolated_edge_groups:
        # 查找每个孤立边的最近邻点
        src_neighbor, dst_neighbor = group[0], group[-1]
        for node in [group[0], group[-1]]:
            point = points[node]

            vectors_to_neighbors = points - point
            distances = np.linalg.norm(vectors_to_neighbors, axis=1)

            valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
            closest_neighbor = None
            closest_distance = float('inf')

            for j in valid_indices:
                if j in group:  # 排除孤立组内的点
                    continue

                # vector_to_neighbor = vectors_to_neighbors[j]
                # vector_norm = np.linalg.norm(vector_to_neighbor)
                #
                # if vector_norm == 0:
                #     continue

                # cosine_similarity = np.dot(direction, vector_to_neighbor) / (direction_norm * vector_norm)
                if distances[j] < closest_distance:
                    closest_neighbor = j
                    closest_distance = distances[j]

            if closest_neighbor is not None:
                if node == group[0]:
                    src_neighbor = closest_neighbor
                else:
                    dst_neighbor = closest_neighbor
                edges.add(tuple(sorted((node, closest_neighbor))))

        # 移除首尾两点近邻点其他路径上的边
        for node in [src_neighbor, dst_neighbor]:
            neighbors = adjacency_list[node]
            for neighbor in neighbors:
                edge = tuple(sorted((node, neighbor)))
                edges.discard(edge)

    # save_as_obj(points, list(edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-2.obj')

    # 统计每个点的连接次数
    point_connection_count = {i: 0 for i in range(points.shape[0])}
    for i, j in edges:
        point_connection_count[i] += 1
        point_connection_count[j] += 1

    # 找到所有孤立点
    isolated_points = {i for i, count in point_connection_count.items() if count == 1}
    # np.savetxt(r'E:\structure_line\data\NerVE\res\outlier.xyz', points[list(isolated_points)])

    edge_dict = {i: [] for i in range(points.shape[0])}
    for i, j in edges:
        edge_dict[i].append(j)
        edge_dict[j].append(i)

    new_edges = set(edges)  # 初始化为已有的边

    # 遍历孤立点时使用列表副本，避免集合大小在迭代中变化
    isolated_points_list = list(isolated_points)

    for i in isolated_points_list:
        if i not in isolated_points:
            continue  # 跳过已经更新的点

        point = points[i]
        neighbors = edge_dict[i]
        if len(neighbors) == 1:  # 仅处理孤立点
            connected_point = points[neighbors[0]]

            # 当前边的方向向量
            edge_vector = connected_point - point
            edge_vector_norm = np.linalg.norm(edge_vector)
            if edge_vector_norm == 0:
                continue

            edge_unit_vector = edge_vector / edge_vector_norm

            # 计算所有其他孤立点到当前边的投影距离
            min_projection_distance = float('inf')
            best_candidate = -1

            for j in list(isolated_points):  # 遍历孤立点的副本
                if j == i or j in neighbors:
                    continue  # 排除自身和已连接点

                candidate_point = points[j]
                vector_to_candidate = candidate_point - point
                distance_to_candidate = np.linalg.norm(vector_to_candidate, axis=-1)

                # 投影距离 = 向量减去其在边方向上的投影
                projection_length = np.dot(vector_to_candidate, edge_unit_vector)
                projection_point = point + projection_length * edge_unit_vector
                distance_to_edge = np.linalg.norm(candidate_point - projection_point)

                if distance_to_edge < min_projection_distance and distance_to_candidate < distance_threshold:
                    min_projection_distance = distance_to_edge
                    best_candidate = j

            # 如果找到最优候选点，则添加边，并更新孤立点状态
            if best_candidate != -1:
                new_edge = tuple(sorted((i, best_candidate)))
                new_edges.add(new_edge)
                # 更新连接状态
                edge_dict[i].append(best_candidate)
                edge_dict[best_candidate].append(i)
                isolated_points.discard(best_candidate)  # 移除目标点的孤立状态
                isolated_points.discard(i)  # 移除当前点的孤立状态

    # save_as_obj(points, list(edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-3.obj')

    for remain_point in isolated_points:
        vectors_to_neighbors = points - points[remain_point]
        distances = np.linalg.norm(vectors_to_neighbors, axis=1)
        valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]

        closest_neighbor = None
        closest_distance = float('inf')

        for j in valid_indices:
            if distances[j] < closest_distance:
                closest_neighbor = j
                closest_distance = distances[j]

        if closest_neighbor is not None:
            new_edges.add(tuple(sorted((remain_point, closest_neighbor))))

    # save_as_obj(points, list(edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-4.obj')

    # 删除度数大于 2 的边
    edge_counts = {i: 0 for i in range(num_points)}
    for edge in new_edges:
        edge_counts[edge[0]] += 1
        edge_counts[edge[1]] += 1

    edges_to_remove = [edge for edge in new_edges if edge_counts[edge[0]] > 2 and edge_counts[edge[1]] > 2]
    for edge in edges_to_remove:
        new_edges.discard(edge)

    return list(new_edges)

def extend_isolated_points_v3(points, edges, distance_threshold=0.08, k=1):
    """
    对所有在边中只出现一次的点，
    计算与当前边投影距离最短的孤立点并将两者连接。

    参数：
    - points: numpy.ndarray，形状为 (N, 3)，点的坐标。
    - edges: list of tuple，初始边的连接关系。

    返回：
    - extended_edges: list of tuple，扩展后的边的连接关系。
    """
    edges = set(edges)

    for edge in edges:
        if edge[0] > edge[1]:
            edges.discard(edge)

    num_points = points.shape[0]

    # 构建邻接表
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    # save_as_obj(points, list(edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1.obj')

    # # 查找点数小于 5 的孤立边
    # def find_isolated_edges():
    #     visited = set()
    #     isolated_edges = []
    #
    #     def dfs(node, path):
    #         if node in visited:
    #             return
    #         visited.add(node)
    #         path.append(node)
    #         for neighbor in adjacency_list[node]:
    #             if neighbor not in visited:
    #                 dfs(neighbor, path)
    #
    #     for i in range(num_points):
    #         if i not in visited:
    #             path = []
    #             dfs(i, path)
    #             if 1 < len(path) < 5:
    #                 isolated_edges.append(path)
    #
    #     return isolated_edges
    #
    # isolated_edge_groups = find_isolated_edges()
    #
    # # 重新连接孤立边
    # for group in isolated_edge_groups:
    #     for node in [group[0], group[-1]]:  # 仅处理首尾点
    #         point = points[node]
    #
    #         neighbors = adjacency_list[node]
    #         connected_point = neighbors[0]
    #         edge_vector = point - points[connected_point]
    #         edge_vector_norm = np.linalg.norm(edge_vector)
    #         if edge_vector_norm == 0:
    #             continue
    #
    #         edge_unit_vector = edge_vector / edge_vector_norm
    #
    #         vectors_to_neighbors = point - points
    #         distances = np.linalg.norm(vectors_to_neighbors, axis=1)
    #
    #         valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
    #         best_candidate = None
    #         max_cosine_similarity = -float('inf')
    #
    #         for j in valid_indices:
    #             if j in group or (node, j) in edges or (j, node) in edges:  # 排除孤立组内的点或已连接点
    #                 continue
    #
    #             vector_to_candidate = points[j] - point
    #             vector_to_candidate_norm = np.linalg.norm(vector_to_candidate)
    #             if vector_to_candidate_norm == 0:
    #                 continue
    #
    #             # 计算余弦相似度
    #             cosine_similarity = np.dot(edge_unit_vector, vector_to_candidate) / vector_to_candidate_norm
    #
    #             if cosine_similarity > max_cosine_similarity:
    #                 best_candidate = j
    #                 max_cosine_similarity = cosine_similarity
    #
    #         if best_candidate is not None:
    #             edges.add(tuple(sorted((node, best_candidate))))
    #             adjacency_list[node].append(best_candidate)
    #             adjacency_list[best_candidate].append(node)

        # 移除首尾两点近邻点其他路径上的边
        # for node in [src_neighbor, dst_neighbor]:
        #     neighbors = adjacency_list[node]
        #     for neighbor in neighbors:
        #         edge = tuple(sorted((node, neighbor)))
        #         edges.discard(edge)

    # save_as_obj(points, list(edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-2.obj')

    # 统计每个点的连接次数
    point_connection_count = {i: 0 for i in range(points.shape[0])}
    for i, j in edges:
        point_connection_count[i] += 1
        point_connection_count[j] += 1

    # 找到所有孤立点
    isolated_points = {i for i, count in point_connection_count.items() if count <= 1}
    # np.savetxt(r'E:\structure_line\data\NerVE\res\outlier.xyz', points[list(isolated_points)])

    edge_dict = {i: [] for i in range(points.shape[0])}
    for i, j in edges:
        edge_dict[i].append(j)
        edge_dict[j].append(i)

    new_edges = set(edges)  # 初始化为已有的边

    for point in list(isolated_points):
        if point not in isolated_points:
            continue  # 跳过已经更新的点

        neighbors = adjacency_list[point]
        if len(neighbors) == 1:  # 孤立点已有一条边
            connected_point = neighbors[0]
            edge_vector = points[point]- points[connected_point]
            edge_vector_norm = np.linalg.norm(edge_vector)
            if edge_vector_norm == 0:
                continue

            edge_unit_vector = edge_vector / edge_vector_norm

            vectors_to_other_points = points[point] - points
            distances = np.linalg.norm(vectors_to_other_points, axis=1)

            valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
            best_candidate = None
            max_cosine_similarity = -float('inf')

            for j in valid_indices:
                if (point, j) in new_edges or (j, point) in new_edges or j in neighbors:  # 排除已连接点
                    continue

                vector_to_candidate = points[j] - points[point]
                candidate_norm = np.linalg.norm(vector_to_candidate)

                if candidate_norm == 0:
                    continue

                cosine_similarity = np.dot(vector_to_candidate, edge_unit_vector) / candidate_norm

                if cosine_similarity > max_cosine_similarity:
                    best_candidate = j
                    max_cosine_similarity = cosine_similarity

            if best_candidate is not None:
                new_edges.add(tuple(sorted((point, best_candidate))))
                adjacency_list[point].append(best_candidate)
                adjacency_list[best_candidate].append(point)
                isolated_points.discard(best_candidate)
                isolated_points.discard(point)

        elif len(neighbors) == 0:  # 度为 0 的孤立点
            vectors_to_other_points = points - points[point]
            distances = np.linalg.norm(vectors_to_other_points, axis=1)

            valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
            closest_neighbor = None
            closest_distance = float('inf')

            for j in valid_indices:
                if (point, j) in new_edges or (j, point) in new_edges:  # 排除已连接点
                    continue

                if distances[j] < closest_distance:
                    closest_neighbor = j
                    closest_distance = distances[j]

            if closest_neighbor is not None:
                new_edges.add(tuple(sorted((point, closest_neighbor))))
                adjacency_list[point].append(closest_neighbor)
                adjacency_list[closest_neighbor].append(point)
                if point_connection_count[closest_neighbor] >= 1:
                    isolated_points.discard(closest_neighbor)


    # save_as_obj(points, list(new_edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-2.obj')
    # while True:
    #     point_connection_count = {i: 0 for i in range(points.shape[0])}
    #     for i, j in new_edges:
    #         point_connection_count[i] += 1
    #         point_connection_count[j] += 1
    #
    #     # 找到所有孤立点
    #     isolated_points = {i for i, count in point_connection_count.items() if count == 1}
    #     if len(isolated_points) == 0:
    #         break
    #     for remain_point in list(isolated_points):
    #         vectors_to_neighbors = points - points[remain_point]
    #         distances = np.linalg.norm(vectors_to_neighbors, axis=1)
    #         valid_indices = np.where((distances > 0))[0]
    #
    #         closest_neighbor = None
    #         closest_distance = float('inf')
    #
    #         for j in valid_indices:
    #             if (remain_point, j) in new_edges or (j, remain_point) in new_edges:  # 检查是否已经连接
    #                 continue
    #             if distances[j] < closest_distance:
    #                 closest_neighbor = j
    #                 closest_distance = distances[j]
    #
    #         if closest_neighbor is not None:
    #             new_edges.add(tuple(sorted((remain_point, closest_neighbor))))
    #         # isolated_points.discard(remain_point)

    # 如果孤立点距离所有其他孤立点过远，连接最近的非孤立点
    edge_counts = {i: 0 for i in range(num_points)}
    for edge in new_edges:
        if edge[0] > edge[1]:
            new_edges.discard(edge)

    for edge in new_edges:
        edge_counts[edge[0]] += 1
        edge_counts[edge[1]] += 1

    isolated_points = [i for i, count in edge_counts.items() if count == 1]
    # np.savetxt(r'E:\structure_line\data\NerVE\res\outlier1.xyz', points[list(isolated_points)])

    for point in isolated_points:
        neighbors = adjacency_list[point]
        if len(neighbors) == 1:  # 孤立点已有一条边
            connected_point = neighbors[0]
            edge_vector = points[point] - points[connected_point]
            edge_vector_norm = np.linalg.norm(edge_vector)
            if edge_vector_norm == 0:
                continue

            edge_unit_vector = edge_vector / edge_vector_norm

            vectors_to_other_points = points[point] - points
            distances = np.linalg.norm(vectors_to_other_points, axis=1)

            valid_indices = np.where((distances > 0) & (distances <= distance_threshold))[0]
            best_candidate = None
            max_cosine_similarity = -float('inf')

            for j in valid_indices:
                if (point, j) in new_edges or (j, point) in new_edges or j in neighbors:  # 排除已连接点
                    continue

                vector_to_candidate = points[j] - points[point]
                candidate_norm = np.linalg.norm(vector_to_candidate)

                if candidate_norm == 0:
                    continue

                # 计算余弦相似度
                cosine_similarity = np.dot(vector_to_candidate, edge_unit_vector) / candidate_norm

                if cosine_similarity > max_cosine_similarity:
                    best_candidate = j
                    max_cosine_similarity = cosine_similarity

            if best_candidate is not None:
                new_edges.add(tuple(sorted((point, best_candidate))))
                adjacency_list[point].append(best_candidate)
                adjacency_list[best_candidate].append(point)


    # save_as_obj(points, list(new_edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-3.obj')

    # 删除度数大于 2 的边
    # edge_counts = {i: 0 for i in range(num_points)}
    # for edge in new_edges:
    #     edge_counts[edge[0]] += 1
    #     edge_counts[edge[1]] += 1
    #
    # edges_to_remove = [edge for edge in new_edges if edge_counts[edge[0]] > 2 and edge_counts[edge[1]] > 2]
    # for edge in edges_to_remove:
    #     new_edges.discard(edge)
    new_edges = remove_high_degree_edges(points, new_edges)

    # save_as_obj(points, list(new_edges), r'E:\structure_line\data\NerVE\res\00000006_pred_v1-4.obj')

    return list(new_edges)

def remove_dual_paths(points, edges):
    """
    删除路径长度为2（两个端点通过一个中间点相连）中的度为2的中间点，
    并选择与中间点相连两条边夹角最小的点进行删除。

    参数：
    - points: numpy.ndarray, 点的坐标。
    - edges: list of tuple, 边的连接关系。

    返回：
    - updated_points: numpy.ndarray, 删除点后的点集合。
    - updated_edges: list of tuple, 删除边后的边集合。
    """
    # 构建邻接表并统计每个点的度
    num_points = points.shape[0]
    degree = {i: 0 for i in range(num_points)}
    # 构建邻接表
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        a, b = edge
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)
        degree[a] += 1
        degree[b] += 1

    points_to_remove = set()  # 记录需要删除的点
    edges_to_remove = set()   # 记录需要删除的边

    # 遍历所有边，查找路径长度为2的子路径
    for a in range(num_points):
        if degree[a] > 2:  # a为度大于2的端点
            max_cosine_val = -float('inf')
            point_to_remove = None
            dst_point = None
            for middle in adjacency_list[a]:
                if degree[middle] == 2 and middle not in points_to_remove:  # 中间点度为2
                    for b in adjacency_list[middle]:
                        if b != a and degree[b] > 2:  # b为另一个端点
                            # 此时 (a - middle - b) 是路径长度为2
                            # 计算夹角，找出夹角最小的两条边
                            vec1 = points[a] - points[middle]
                            vec2 = points[b] - points[middle]
                            cosine_angle = calculate_angle(vec1, vec2)

                            if cosine_angle > max_cosine_val:
                                max_cosine_val = cosine_angle
                                point_to_remove = middle
                                dst_point = b

                            # 如果夹角最小，标记中间点为删除点
            if point_to_remove is not None and dst_point is not None:
                points_to_remove.add(point_to_remove)
                edges_to_remove.add(tuple(sorted((a, point_to_remove))))
                edges_to_remove.add(tuple(sorted((point_to_remove, dst_point))))
                adjacency_list[a].remove(point_to_remove)
                adjacency_list[dst_point].remove(point_to_remove)
                del adjacency_list[point_to_remove]

    # 更新点集与边集
    new_points = []
    new_index_mapping = {}
    updated_edges = set()
    edges_set = set([edge for edge in edges if edge not in edges_to_remove and edge[::-1] not in edges_to_remove])

    # 新点索引映射
    for i in range(num_points):
        if i not in points_to_remove:
            new_index_mapping[i] = len(new_points)
            new_points.append(points[i])

    # 更新edges和邻接表
    for edge in edges_set:
        if edge[0] not in points_to_remove and edge[1] not in points_to_remove:
            updated_edges.add((new_index_mapping[edge[0]], new_index_mapping[edge[1]]))

    # 返回更新后的points和edges
    updated_points = np.array(new_points)  # 更新points
    # print(updated_points.shape[0], len(updated_edges))
    return updated_points, updated_edges

def find_and_remove_triangular_loops(points, edges):
    """
    找出只有三个点的环路，并删除环路中最长的边。如果环路中的点在删除后度为1，
    则与非环路中的最近邻点相连。

    参数：
    - points: numpy.ndarray, 点的坐标。
    - edges: list of tuple, 边的连接关系。

    返回：
    - updated_points: numpy.ndarray, 删除边后的点集合。
    - updated_edges: list of tuple, 删除边后的边集合。
    """
    # 构建邻接表
    num_points = points.shape[0]
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        a, b = edge
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)

    edges = set(edges)

    # 找到所有三点环路
    triangles = []
    for a in range(num_points):
        for b in adjacency_list[a]:
            if b > a:  # 避免重复
                for c in adjacency_list[b]:
                    if c > b and a in adjacency_list[c]:
                        triangles.append((a, b, c))

    # 删除环路中最长的边
    for triangle in triangles:
        a, b, c = triangle
        edges_in_triangle = [(a, b), (b, c), (c, a)]
        edge_lengths = {
            (a, b): np.linalg.norm(points[a] - points[b]),
            (b, c): np.linalg.norm(points[b] - points[c]),
            (c, a): np.linalg.norm(points[c] - points[a]),
        }

        # 找到最长的边
        longest_edge = max(edge_lengths, key=edge_lengths.get)
        edges.discard(tuple(sorted(longest_edge)))

        # 更新邻接表
        adjacency_list[longest_edge[0]].remove(longest_edge[1])
        adjacency_list[longest_edge[1]].remove(longest_edge[0])

    # 检查删除后是否有点的度为1，若有则与最近非环路点相连
    for node in range(num_points):
        if len(adjacency_list[node]) == 1:
            neighbor = adjacency_list[node][0]
            triangle_neighbors = []
            for loop in triangles:
                if node in loop:
                    triangle_neighbors += [item for item in loop if item != node]
            min_distance = float('inf')
            nearest_point = None

            for i in range(num_points):
                if i != node and i != neighbor and i not in triangle_neighbors:
                    distance = np.linalg.norm(points[node] - points[i])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = i

            if nearest_point is not None:
                new_edge = tuple(sorted((node, nearest_point)))
                edges.add(new_edge)
                adjacency_list[node].append(nearest_point)
                adjacency_list[nearest_point].append(node)

    # 返回更新后的点集和边集
    return points, list(edges)


def find_intersection(p1, p2, p3, p4):
    """
    判断两条线段是否在三维空间中相交，并计算交点。

    参数：
    - p1, p2: 第一条线段的两个端点
    - p3, p4: 第二条线段的两个端点

    返回：
    - intersect: bool，是否相交
    - intersection_point: np.ndarray，相交点的坐标（如果存在）
    """

    def is_point_on_segment(pt, seg_start, seg_end):
        """判断点是否在线段上"""
        return np.allclose(np.linalg.norm(pt - seg_start) + np.linalg.norm(pt - seg_end),
                           np.linalg.norm(seg_end - seg_start))

    # 构造线段向量
    v1 = p2 - p1
    v2 = p4 - p3

    # 求解交点
    cross_product = np.cross(v1, v2)
    denom = np.linalg.norm(cross_product) ** 2

    if np.isclose(denom, 0):
        return False, None  # 平行或共线

    t = np.dot(np.cross(p3 - p1, v2), cross_product) / denom
    u = np.dot(np.cross(p3 - p1, v1), cross_product) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = p1 + t * v1
        if is_point_on_segment(intersection_point, p1, p2) and is_point_on_segment(intersection_point, p3, p4):
            return True, intersection_point

    return False, None


def resolve_edge_intersections(points, edges, distance_threshold=0.05):
    """
    处理边的交叉情况，仅考虑近邻边。

    参数：
    - points: numpy.ndarray, 点的坐标。
    - edges: list of tuple, 边的连接关系。
    - distance_threshold: float, 近邻边的最大距离阈值。

    返回：
    - updated_edges: list of tuple, 更新后的边集合。
    """
    edges = set(edges)
    num_points = points.shape[0]

    # 构建邻接表
    adjacency_list = {i: [] for i in range(num_points)}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    edges_to_remove = set()
    edges_to_add = set()

    # 检查所有边的组合
    for edge1, edge2 in combinations(edges, 2):
        # 提取边的端点
        a, b = edge1
        c, d = edge2

        # 排除共端点的边
        if len(set([a, b, c, d])) < 4:
            continue

        # 距离过滤（近邻检查）
        dist_ac = np.linalg.norm(points[a] - points[c])
        dist_ad = np.linalg.norm(points[a] - points[d])
        dist_bc = np.linalg.norm(points[b] - points[c])
        dist_bd = np.linalg.norm(points[b] - points[d])
        if (dist_ac > distance_threshold and dist_ad> distance_threshold and
                dist_bc> distance_threshold and dist_bd > distance_threshold):
            continue

        # 检查交点
        intersect, intersection_point = find_intersection(points[a], points[b], points[c], points[d])
        if intersect:
            # 删除原有的两条边
            edges_to_remove.add(edge1)
            edges_to_remove.add(edge2)

            # 找到交叉边的最近点，并与交点相连
            for idx in [index for index, value in sorted(enumerate([dist_ac, dist_ad, dist_bc, dist_bd]), key=lambda x: x[1])][:3]:
                if idx == 0:
                    edges_to_add.add(tuple(sorted((a, c))))
                elif idx == 1:
                    edges_to_add.add(tuple(sorted((a, d))))
                elif idx == 2:
                    edges_to_add.add(tuple(sorted((b, c))))
                else:
                    edges_to_add.add(tuple(sorted((b, d))))
            # closest_points = sorted([a, b, c, d], key=lambda idx: np.linalg.norm(points[idx] - intersection_point))
            # edges_to_add.add(tuple(sorted((closest_points[0], closest_points[1]))))
            # edges_to_add.add(tuple(sorted((closest_points[2], closest_points[3]))))

    # 更新边集合
    edges = edges - edges_to_remove
    edges = edges | edges_to_add

    print(f'Removed {len(edges_to_remove)} edges')

    return list(edges)

def insert_points(points, edges, num_insert=5):
    """
    Insert new points between each pair of connected points.

    Args:
        points (numpy.ndarray): N*3 array of points.
        edges (list of tuples): List of edges as (v1, v2) indices.
        num_insert (int): Number of points to insert between each pair of points.

    Returns:
        numpy.ndarray: New array of points including the inserted ones.
        list of tuples: Updated list of edges connecting all points.
    """
    new_points = []
    new_edges = []

    # Keep track of the original number of points
    point_offset = len(points)

    for i, (v1, v2) in enumerate(edges):
        p1, p2 = points[v1], points[v2]

        # Generate interpolated points between p1 and p2
        for j in range(1, num_insert + 1):
            t = j / (num_insert + 1)
            new_point = (1 - t) * p1 + t * p2
            new_points.append(new_point)

            # Connect new points to the previous and next points
            if j == 1:  # Connect the first interpolated point to p1
                new_edges.append((v1, point_offset))
            else:  # Connect subsequent interpolated points to the previous interpolated point
                new_edges.append((point_offset - 1, point_offset))

            # Connect the last interpolated point to p2
            if j == num_insert:
                new_edges.append((point_offset, v2))

            point_offset += 1

    # Combine original points with new points
    all_points = np.vstack([points, np.array(new_points)])

    # Include original edges
    updated_edges = edges + new_edges

    return all_points, updated_edges


def read_obj(filepath):
    vertices = []
    edges = []

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':  # 顶点数据
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)

            elif parts[0] == 'l':  # 边线定义 (用于 OBJ 文件的线段)
                # OBJ 的边索引从 1 开始，需要减 1 转为 0 索引
                edge = tuple(map(lambda x: int(x) - 1, parts[1:]))
                if len(edge) == 2:  # 确保是有效的边
                    edges.append(edge)

            elif parts[0] == 'f':  # 面 (用于从面推导边)
                # OBJ 的面定义是基于索引的，通常格式如 "f 1 2 3"
                indices = list(map(lambda x: int(x.split('/')[0]) - 1, parts[1:]))
                for i in range(len(indices)):
                    edge = (indices[i], indices[(i + 1) % len(indices)])  # 闭合边
                    if edge not in edges and (edge[1], edge[0]) not in edges:
                        edges.append(edge)

    return np.array(vertices), edges

def read_obj_v2(filepath):
    vertices = []  # 存储顶点
    edges = []     # 存储边
    faces = []     # 存储面

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            # 读取顶点
            if parts[0] == 'v':  # 顶点定义
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)

            # 读取边
            elif parts[0] == 'l':  # 直接定义的边
                edge = tuple(map(lambda x: int(x) - 1, parts[1:]))  # 索引从 1 开始，需要减 1
                edges.append(edge)

            # 读取面
            elif parts[0] == 'f':  # 面定义
                # 分割每个顶点索引，OBJ 面可能包含法线或纹理坐标，用 "/" 处理
                face = [int(v.split('/')[0]) - 1 for v in parts[1:]]
                faces.append(face)

    return vertices, edges, faces

def rescale_sketch(target_bb_diagonal_length, mesh_folder):
    input_mesh_path = os.path.join(mesh_folder, 'cad_00000003.obj')
    input_sketch_path = os.path.join(mesh_folder, 'mesh_06.obj')
    V, lines = read_obj(input_mesh_path)

    initial_bb_diag_length = np.linalg.norm(np.max(V, axis=0) - np.min(V, axis=0))

    scale_factor = target_bb_diagonal_length / initial_bb_diag_length

    V = V * scale_factor

    center = np.mean(V, axis=0)

    # center[0] = 0

    V = V - center
    print(f"Scaling by factor = {scale_factor}, translating by: {-center}")

    save_as_obj(V, lines, os.path.join(mesh_folder, 'cad_00000003_norm.obj'))
    print('Saved mesh')

    # V, lines = read_obj(input_sketch_path)
    V, lines, faces = read_obj_v2(input_sketch_path)
    V = np.array(V)
    V = V * scale_factor
    V = V - center

    save_as_obj_v2(os.path.join(mesh_folder, 'mesh_06_norm.obj'), V.tolist(), lines, faces)


def save_as_obj(points, edges, filename):
    """
    保存点和边到 .obj 文件。

    参数：
    - points: [n, 3]
    - edges: [n, 2]
    - filename: str，输出文件的名称。
    """
    with open(filename, 'w') as file:
        # 写入顶点
        for x, y, z in points:
            file.write(f"v {x} {y} {z}\n")

        # 写入边
        for i, j in edges:
            file.write(f"l {i + 1} {j + 1}\n")  # .obj 文件索引从 1 开始


def save_as_obj_v2(filepath, vertices, edges=None, faces=None):
    """
    将顶点、边和面写入 .obj 文件。

    参数：
    - filepath: 输出文件路径
    - vertices: 顶点列表，形如 [[x1, y1, z1], [x2, y2, z2], ...]
    - edges: 边列表，形如 [(v1, v2), (v2, v3), ...]，可选
    - faces: 面列表，形如 [[v1, v2, v3], [v4, v5, v6], ...]，可选
    """
    with open(filepath, 'w') as file:
        # 写入顶点
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # 写入边（如果提供）
        if edges:
            for edge in edges:
                # .obj 文件的索引从 1 开始，因此需加 1
                file.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

        # 写入面（如果提供）
        if faces:
            for face in faces:
                # .obj 文件的索引从 1 开始，因此需加 1
                face_indices = " ".join(str(idx + 1) for idx in face)
                file.write(f"f {face_indices}\n")


# gt_name = r'E:\structure_line\data\NerVE\res\00000003.pkl'
#
# with open(gt_name, 'rb') as f:
#     data = pickle.load(f)
#
# pc = data["points"]
# edges = data["edges"]
#
# save_as_obj(pc, edges, r'E:\structure_line\data\NerVE\res\00000003_gt.obj')

# pred_name = r'E:\structure_line\data\NerVE\res\00001108_pred.txt'
# save_path = r'E:\structure_line\data\NerVE\res\00001108_pred.obj'
#
# data = np.loadtxt(pred_name)
# pc = data[:, :3]
# direction = data[:, 3:]
#
# # edges = connect_edges(pc, direction)
# edges = connect_edges_v4(pc, direction)
# # save_as_obj(pc, edges, save_path)
# # edges = extend_isolated_points(pc, direction, edges, extended_distance_threshold=0.2, extended_angle_threshold=0.8)
# pc, edges = remove_outliers(pc, edges)
# edges = extend_isolated_points_v3(pc, edges)
# pc, edges = remove_dual_paths(pc, edges)
# pc, edges = find_and_remove_triangular_loops(pc, edges)
# edges = resolve_edge_intersections(pc, edges)
# save_as_obj(pc, edges, save_path)

# edge_path = r'E:\structure_line\data\NerVE\res\00001108_pred.obj'
# save_path = r'E:\structure_line\result\reconstruction\cad_1108_dense_edge.obj'
#
# pc, edges = read_obj(edge_path)
# dense_pc, dense_edges = insert_points(pc, edges, num_insert=1)
# # np.savetxt(save_path, dense_pc)
# save_as_obj(dense_pc, dense_edges, save_path)

# mesh_folder = r'E:\structure_line\result\tmp'
# rescale_sketch(1.0, mesh_folder)

pred_root = './exp/nerve_v5/semseg-pt-v3m1-0-base/result'
save_root = './exp/curve'
os.makedirs(save_root, exist_ok=True)

num = 0
total_times = 0
for file_name in os.listdir(pred_root):
    model_id = file_name.split('_')[0]
    # if model_id != '00000068' and model_id != '00001108':
    #     continue
    print(model_id)
    file_path = os.path.join(pred_root, file_name)
    data = np.loadtxt(file_path)
    pc = data[:, :3]
    direction = data[:, 3:]
    start = time.time()
    edges = connect_edges_v4(pc, direction)
    pc, edges = remove_outliers(pc, edges)
    edges = extend_isolated_points_v3(pc, edges)
    pc, edges = remove_dual_paths(pc, edges)
    pc, edges = find_and_remove_triangular_loops(pc, edges)
    # edges = resolve_edge_intersections(pc, edges)
    res = {
        'points': pc,
        'edges': edges
    }
    end = time.time()
    total_times += (end - start)
    pkl_save_dir = os.path.join(save_root, 'pkl', model_id)
    obj_save_dir = os.path.join(save_root, 'obj', model_id)
    os.makedirs(pkl_save_dir, exist_ok=True)
    os.makedirs(obj_save_dir, exist_ok=True)
    with open(os.path.join(pkl_save_dir, 'pred_pwl_curve.pkl'), 'wb') as f:
        pickle.dump(res, f)
    save_as_obj(pc, edges, os.path.join(obj_save_dir, 'pred_curve.obj'))
    num += 1
print(f'Processed {num} models')
print(f'Time cost: {total_times / num}')

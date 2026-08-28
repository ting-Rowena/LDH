import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from CellFateLandscape import NonEquilibriumCellFateLandscape
import cycler

custom_colors = ['#0E4DA6', '#00AE75', '#FFB300', '#E02F62', '#EBE800', '#6067B6','#009C94', '#FEB8D5']
plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=custom_colors)


class LandscapeVisualizer:
    """景观可视化工具"""

    def __init__(self, landscape_analyzer):
        self.analyzer = landscape_analyzer
        # 定义颜色循环，用于区分不同路径

        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_landscape_with_path(self, path_result, use_3d=False, show_flux=True, figsize=(6, 5), save_path=None):
        """绘制景观和最优路径"""

        if use_3d:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            self._plot_3d_landscape(ax, path_result, show_flux)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_2d_landscape(ax, path_result, show_flux)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig, ax

    def _plot_2d_landscape(self, ax, path_result, show_flux):
        """绘制2D景观"""
        # 创建网格
        x_min, x_max = self.analyzer.cell_positions_2d[:, 0].min(), self.analyzer.cell_positions_2d[:, 0].max()
        y_min, y_max = self.analyzer.cell_positions_2d[:, 1].min(), self.analyzer.cell_positions_2d[:, 1].max()

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 计算网格势能
        Z = self.analyzer.U_func_2d(grid_points).reshape(xx.shape)

        # 绘制等高线
        contourf = ax.contourf(xx, yy, Z, levels=50, alpha=0.7, cmap='coolwarm')
        # contour = ax.contour(xx, yy, Z, levels=20, colors='k', alpha=0.3, linewidths=0.5)

        # 绘制路径
        path = path_result['path']
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=3, label='Optimal Path')

        # 路径颜色表示作用量
        # scatter = ax.scatter(path[:, 0], path[:, 1], c=path_result['action'],cmap='plasma', s=50, alpha=0.8, zorder=5)

        # 标记过渡态
        ts_idx = path_result['transition_state_idx']
        ax.scatter(path[ts_idx, 0], path[ts_idx, 1],
                   c='yellow', s=100, marker='o', edgecolors='black',
                   linewidth=2, label='Transition State', zorder=6)

        # 标记细胞状态
        if hasattr(self.analyzer, 'cell_states_2d'):
            for state_id, state_info in self.analyzer.cell_states_2d.items():
                pos = state_info['position']
                ax.scatter(pos[0], pos[1], c='red', s=100, marker='s',
                           edgecolors='red', linewidth=2, zorder=7)
                ax.text(pos[0], pos[1] + 0.5, state_id, fontsize=12,
                        ha='center', va='bottom', color='white', weight='bold', zorder=8)

        # 绘制通量场（如果要求）
        if show_flux and hasattr(self.analyzer, 'v_func_2d'):
            # 在稀疏网格上绘制通量箭头
            x_flux = np.linspace(x_min, x_max, 15)
            y_flux = np.linspace(y_min, y_max, 15)
            xx_flux, yy_flux = np.meshgrid(x_flux, y_flux)
            flux_points = np.c_[xx_flux.ravel(), yy_flux.ravel()]

            flux_vectors = np.array([self.analyzer.v_func_2d([p])[0] for p in flux_points])

            ax.quiver(flux_points[:, 0], flux_points[:, 1],
                      flux_vectors[:, 0], flux_vectors[:, 1],
                      scale=20, color='red', alpha=0.6, width=0.005)

        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.set_title('2D Cell Fate Landscape with Optimal Path\n'
                     f'Transition State at point {ts_idx}, Total Action: {path_result["total_action"]:.4f}', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(labelsize=8)

        # 添加颜色条
        #plt.colorbar(contourf, ax=ax, label='Potential Energy')
        # plt.colorbar(scatter, ax=ax, label='Action along path')
        cbar = plt.colorbar(contourf, ax=ax)
        # 设置颜色条标签（Potential Energy）的字体大小
        cbar.set_label('Potential Energy', fontsize=9)
        # 设置颜色条刻度数值的字体大小（核心：调整刻度字体）
        cbar.ax.tick_params(labelsize=8)

    def _plot_3d_landscape(self, ax, path_result, show_flux):
        """绘制3D景观"""
        # 使用散点图显示3D景观
        scatter = ax.scatter(self.analyzer.cell_positions_3d[:, 0],
                             self.analyzer.cell_positions_3d[:, 1],
                             self.analyzer.cell_positions_3d[:, 2],
                             c=self.analyzer.potential_energy,
                             cmap='coolwarm', alpha=0.3, s=10)

        # 绘制3D路径
        path = path_result['path']
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=4, label='Optimal Path')

        # 路径颜色表示作用量
        path_scatter = ax.scatter(path[:, 0], path[:, 1], path[:, 2],
                                  c=path_result['action'], cmap='plasma',
                                  s=20, alpha=0.9, depthshade=True)

        # 标记过渡态
        ts_idx = path_result['transition_state_idx']
        ts_pos = path[ts_idx]
        ax.scatter(ts_pos[0], ts_pos[1], ts_pos[2],
                   c='yellow', s=200, marker='o', edgecolors='black',
                   linewidth=2, label='Transition State')

        # 标记细胞状态
        if hasattr(self.analyzer, 'cell_states_3d'):
            for state_id, state_info in self.analyzer.cell_states_3d.items():
                pos = state_info['position']
                ax.scatter(pos[0], pos[1], pos[2],
                           c='red', s=200, marker='o',
                           edgecolors='white', linewidth=2)
                ax.text(pos[0], pos[1], pos[2], state_id, fontsize=12,
                        ha='center', va='bottom', color='white', weight='bold')

        ax.set_xlabel('UMAP 1', labelsize=9)
        ax.set_ylabel('UMAP 2', labelsize=9)
        ax.set_zlabel('UMAP 3', labelsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.set_title('3D Cell Fate Landscape with Optimal Path\n' f'Transition State at point {ts_idx}', labelsize=10)
        ax.legend(labelsize=8)
        ax.view_init(elev=30, azim=45)

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Potential Energy', shrink=0.5)
        plt.colorbar(path_scatter, ax=ax, label='Action along path', shrink=0.5)

    def plot_landscape_with_multiple_paths(self, all_path_results, use_3d=False, show_flux=True, figsize=(6, 5), save_path=None):
        """绘制景观和多个最优路径"""

        if use_3d:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            self._plot_3d_landscape_multiple_paths(ax, all_path_results, show_flux)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_2d_landscape_multiple_paths(ax, all_path_results, show_flux)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig, ax

    def _plot_2d_landscape_multiple_paths(self, ax, all_path_results, show_flux):
        """绘制2D景观和多个路径"""
        # 创建网格
        x_min, x_max = self.analyzer.cell_positions_2d[:, 0].min(), self.analyzer.cell_positions_2d[:, 0].max()
        y_min, y_max = self.analyzer.cell_positions_2d[:, 1].min(), self.analyzer.cell_positions_2d[:, 1].max()

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 计算网格势能
        Z = self.analyzer.U_func_2d(grid_points).reshape(xx.shape)

        # 绘制等高线
        contourf = ax.contourf(xx, yy, Z, levels=50, alpha=0.7, cmap='coolwarm')
        # contour = ax.contour(xx, yy, Z, levels=20, colors='k', alpha=0.3, linewidths=0.5)

        # 绘制多条路径
        legend_handles = []
        color_idx = 0

        for path_key, path_result in all_path_results.items():
            path = path_result['path']
            color = self.color_cycle[color_idx % len(self.color_cycle)]

            # 绘制路径线
            line = ax.plot(path[:, 0], path[:, 1], '-', linewidth=2, color=color, label=path_key, alpha=0.8)[0]
            legend_handles.append(line)

            # 路径颜色表示作用量
            # scatter = ax.scatter(path[:, 0], path[:, 1], c=path_result['action'], cmap='plasma', s=30, alpha=0.6, zorder=5)

            # 标记过渡态
            ts_idx = path_result['transition_state_idx']
            #ax.scatter(path[ts_idx, 0], path[ts_idx, 1], c=color, s=100, marker='o', edgecolors='black', linewidth=1.5, alpha=0.8, zorder=6)
            ax.scatter(path[ts_idx, 0], path[ts_idx, 1], c='red', s=20, marker='o', edgecolors='red', linewidth=1.5,alpha=0.8, zorder=6)

            color_idx += 1

        # 标记细胞状态
        if hasattr(self.analyzer, 'cell_states_2d'):
            for state_id, state_info in self.analyzer.cell_states_2d.items():
                pos = state_info['position']
                #ax.scatter(pos[0], pos[1], c='red', s=100, marker='s', edgecolors='red', linewidth=2, zorder=7)
                ax.scatter(pos[0], pos[1], c='black', s=20, marker='s', edgecolors='black', linewidth=1, zorder=7)
                ax.text(pos[0], pos[1]+0.5, state_id, fontsize=10, ha='center', va='bottom', color='black', zorder=8)

        # 绘制通量场（如果要求）
        if show_flux and hasattr(self.analyzer, 'v_func_2d'):
            # 在稀疏网格上绘制通量箭头
            x_flux = np.linspace(x_min, x_max, 15)
            y_flux = np.linspace(y_min, y_max, 15)
            xx_flux, yy_flux = np.meshgrid(x_flux, y_flux)
            flux_points = np.c_[xx_flux.ravel(), yy_flux.ravel()]

            flux_vectors = np.array([self.analyzer.v_func_2d([p])[0] for p in flux_points])

            ax.quiver(flux_points[:, 0], flux_points[:, 1],
                      flux_vectors[:, 0], flux_vectors[:, 1],
                      scale=20, color='red', alpha=0.6, width=0.005)

        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title('2D Cell Fate Landscape with Multiple Optimal Paths', fontsize=10)

        # 添加图例
        ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

        cbar = plt.colorbar(contourf, ax=ax)
        # 设置颜色条标签（Potential Energy）的字体大小
        cbar.set_label('Potential Energy', fontsize=9)
        # 设置颜色条刻度数值的字体大小（核心：调整刻度字体）
        cbar.ax.tick_params(labelsize=8)


        # 显示路径统计信息
        path_info = "\n".join([f"{key}: Action={result['total_action']:.4f}" for key, result in all_path_results.items()])
        # ax.text(0.02, 0.98, path_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_3d_landscape_multiple_paths(self, ax, all_path_results, show_flux):
        """绘制3D景观和多个路径"""
        # 使用散点图显示3D景观
        scatter = ax.scatter(self.analyzer.cell_positions_3d[:, 0],
                             self.analyzer.cell_positions_3d[:, 1],
                             self.analyzer.cell_positions_3d[:, 2],
                             c=self.analyzer.potential_energy,
                             cmap='viridis', alpha=0.3, s=10)

        # 绘制多条3D路径
        color_idx = 0

        for path_key, path_result in all_path_results.items():
            path = path_result['path']
            color = self.color_cycle[color_idx % len(self.color_cycle)]

            # 绘制3D路径线
            ax.plot(path[:, 0], path[:, 1], path[:, 2], '-', linewidth=3, color=color, label=path_key, alpha=0.8)

            # 路径颜色表示作用量
            path_scatter = ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=path_result['action'], cmap='plasma', s=30,alpha=0.7, depthshade=True)

            # 标记过渡态
            ts_idx = path_result['transition_state_idx']
            ts_pos = path[ts_idx]
            ax.scatter(ts_pos[0], ts_pos[1], ts_pos[2],
                       c=color, s=100, marker='o', edgecolors='black',
                       linewidth=1.5, alpha=0.8)

            color_idx += 1

        # 标记细胞状态
        if hasattr(self.analyzer, 'cell_states_3d'):
            for state_id, state_info in self.analyzer.cell_states_3d.items():
                pos = state_info['position']
                ax.scatter(pos[0], pos[1], pos[2],
                           c='red', s=100, marker='s',
                           edgecolors='red', linewidth=2)
                ax.text(pos[0], pos[1], pos[2], state_id, fontsize=12,
                        ha='center', va='bottom', color='white', weight='bold')

        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.set_zlabel('UMAP 3', fontsize=9)
        ax.set_title('3D Cell Fate Landscape with Multiple Optimal Paths', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.legend(fontsize=8)
        ax.view_init(elev=30, azim=45)

        # 添加颜色条
        # plt.colorbar(scatter, ax=ax, label='Potential Energy', shrink=0.5)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        # 设置颜色条标签（Potential Energy）的字体大小
        cbar.set_label('Potential Energy', fontsize=9)
        # 设置颜色条刻度数值的字体大小（核心：调整刻度字体）
        cbar.ax.tick_params(labelsize=8)


def analyze_cell_fate_transition(adata, start_state, end_state,clustering_key='stage', use_3d=False, n_path_points=100):
    """
    完整的细胞命运转换分析流程

    Parameters:
    -----------
    adata : AnnData
        单细胞数据
    start_state, end_state : str
        起始和结束状态
    clustering_key : str
        聚类键名
    use_3d : bool
        是否使用3D分析
    n_path_points : int
        路径点数
    """

    print("=== 非平衡细胞命运景观分析 ===")

    # 初始化分析器
    analyzer = NonEquilibriumCellFateLandscape(
        adata,
        potential_key='potential',
        embedding_2d_key='X_umap',
        embedding_3d_key='X_umap_3d'
    )

    # 识别细胞状态
    cell_states = analyzer.identify_cell_states(
        clustering_key=clustering_key,
        use_3d=use_3d
    )

    if start_state not in cell_states:
        raise ValueError(f"起始状态 {start_state} 未找到。可用状态: {list(cell_states.keys())}")
    if end_state not in cell_states:
        raise ValueError(f"结束状态 {end_state} 未找到。可用状态: {list(cell_states.keys())}")

    start_pos = cell_states[start_state]['position']
    end_pos = cell_states[end_state]['position']

    print(f"分析转换: {start_state} -> {end_state}")
    print(f"起始位置: {start_pos}")
    print(f"结束位置: {end_pos}")

    # 计算最小作用路径
    print("计算最小作用路径...")
    path_result = analyzer.compute_least_action_path(
        start_pos, end_pos,
        n_points=n_path_points,
        use_3d=use_3d
    )

    if not path_result['success']:
        print("警告: 路径优化未完全收敛")

    # 计算熵产生率
    epr = analyzer.compute_entropy_production_rate(
        path_result['path'],
        use_3d=use_3d
    )

    print(f"\n=== 分析结果 ===")
    print(f"总作用量: {path_result['total_action']:.6f}")
    print(f"过渡态位置: 路径点 {path_result['transition_state_idx']}")
    print(f"过渡态势能: {path_result['potential'][path_result['transition_state_idx']]:.4f}")
    print(f"平均熵产生率: {epr:.6f}")

    # 可视化
    visualizer = LandscapeVisualizer(analyzer)
    fig, ax = visualizer.plot_landscape_with_path(
        path_result,
        use_3d=use_3d,
        show_flux=True
    )

    return {
        'analyzer': analyzer,
        'path_result': path_result,
        'epr': epr,
        'cell_states': cell_states,
        'visualizer': visualizer
    }


def analyze_multiple_cell_fate_transitions(adata, path_pairs, clustering_key='stage', use_3d=False, n_path_points=100, smooth=None):
    """
    分析多个细胞命运转换路径

    Parameters:
    -----------
    adata : AnnData
        单细胞数据
    path_pairs : list of tuples
        状态对列表，例如 [('IIIC', 'IVB'), ('IIIC', 'IIIA')]
    clustering_key : str
        聚类键名
    use_3d : bool
        是否使用3D分析
    n_path_points : int
        路径点数
    """

    print("=== 多路径非平衡细胞命运景观分析 ===")
    print(f"分析路径: {path_pairs}")

    # 初始化分析器
    analyzer = NonEquilibriumCellFateLandscape(
        adata,
        potential_key='potential',
        embedding_2d_key='X_umap',
        embedding_3d_key='X_umap_3d'
    )

    # 识别细胞状态
    cell_states = analyzer.identify_cell_states(
        clustering_key=clustering_key,
        use_3d=use_3d
    )

    # 验证所有状态都存在
    valid_pairs = []
    for start_state, end_state in path_pairs:
        if start_state not in cell_states:
            print(f"错误: 起始状态 {start_state} 未找到")
            continue
        if end_state not in cell_states:
            print(f"错误: 结束状态 {end_state} 未找到")
            continue
        valid_pairs.append((start_state, end_state))

    if not valid_pairs:
        raise ValueError("没有找到有效的状态对")

    # 计算所有路径
    print("计算所有最小作用路径...")
    all_path_results = analyzer.compute_multiple_paths(
        valid_pairs,
        use_3d=use_3d,
        n_points=n_path_points
    )

    # 计算每个路径的熵产生率
    for path_key, path_result in all_path_results.items():
        epr = analyzer.compute_entropy_production_rate(
            path_result['path'],
            use_3d=use_3d
        )
        path_result['epr'] = epr
        if smooth:
            path_result['path'] = smooth_path(
                path_result['path'],
                smoothing_factor=0.1  # 调整平滑因子
            )

    # 打印结果摘要
    print(f"\n=== 多路径分析结果 ===")
    for path_key, path_result in all_path_results.items():
        print(f"路径 {path_key}:")
        print(f"  总作用量: {path_result['total_action']:.6f}")
        print(f"  过渡态位置: 路径点 {path_result['transition_state_idx']}")
        print(f"  过渡态势能: {path_result['potential'][path_result['transition_state_idx']]:.4f}")
        print(f"  平均熵产生率: {path_result['epr']:.6f}")
        print()

    # 可视化
    visualizer = LandscapeVisualizer(analyzer)
    fig, ax = visualizer.plot_landscape_with_multiple_paths(
        all_path_results,
        use_3d=use_3d,
        show_flux=True
    )

    return {
        'analyzer': analyzer,
        'all_path_results': all_path_results,
        'cell_states': cell_states,
        'visualizer': visualizer,
        'figure': fig,  # 返回图形对象
        'axis':ax
    }


def smooth_path(path, smoothing_factor=0.2):
    """使用Savitzky-Golay滤波器平滑路径"""
    from scipy.signal import savgol_filter

    """
    参数：
        path: np.array，形状(N, D)，N为路径点数量，D为维度
        smoothing_factor: float，0~1之间，平滑因子（越大越平滑）
                          控制窗口长度=max(3, 奇数)，避免窗口过小/过大
    返回：
        smoothed_path: np.array，平滑后的路径
    """
    # 校验输入
    smoothed_path = np.zeros_like(path)
    for i in range(path.shape[1]):  # 对每个维度进行平滑
        smoothed_path[:, i] = savgol_filter(
            path[:, i],
            window_length=min(15, len(path)),  # 窗口长度
            polyorder=3,  # 多项式阶数
            mode='interp'
        )
    return smoothed_path
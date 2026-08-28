"""
Heuristic pioneer-gene ranking along embedding LAP paths.

Scores are exploratory rankings (not formal statistical significance unless
empirical_p_value from permutation is reported).
"""

import warnings

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.stats import zscore, percentileofscore
from CellFateLandscape import NonEquilibriumCellFateLandscape
from LandscapeVisualizer import LandscapeVisualizer

DEFAULT_TF_PATHS = ("./allTFs_hg38.txt", "./allTFs_mm.txt")


class PioneerGeneIdentifier:
    """
    基于文章中过渡态和成核机制的关键基因识别
    识别先锋基因作为细胞命运决策的成核种子
    """

    DEFAULT_TF_PATHS = DEFAULT_TF_PATHS

    def __init__(
        self,
        landscape_analyzer,
        tf_gene_paths=None,
        min_expression=0.1,
        n_permutations=100,
        smooth_window=3,
    ):
        """
        Parameters
        ----------
        landscape_analyzer : NonEquilibriumCellFateLandscape
        tf_gene_paths : sequence of str, optional
            Paths to TF gene list files (loaded once at init).
        min_expression : float
            Minimum mean path expression for ranking.
        n_permutations : int
            Permutation count for empirical p-values (0 to skip).
        smooth_window : int
            Moving-average window for path expression smoothing.
        """
        self.analyzer = landscape_analyzer
        self.adata = landscape_analyzer.adata
        self.min_expression = float(min_expression)
        self.n_permutations = int(n_permutations)
        self.smooth_window = max(1, int(smooth_window))
        self.tf_genes = self._load_tf_genes(
            tf_gene_paths if tf_gene_paths is not None else self.DEFAULT_TF_PATHS
        )
        self._all_gene_means = self._compute_all_gene_means()

    def identify_pioneer_genes_along_path(
        self,
        path_result,
        transition_window=5,
        top_n_genes=20,
        use_3d=False,
        n_permutations=None,
    ):
        """
        沿最优路径识别先锋基因（heuristic ranking + optional permutation p-value）。

        Returns
        -------
        dict with pioneer_genes ranked by composite score (not raw validation multiplier).
        """
        transition_idx = path_result["transition_state_idx"]
        n_perm = self.n_permutations if n_permutations is None else int(n_permutations)

        print(f"识别路径上的先锋基因，过渡态位置: {transition_idx}")
        print(
            "注意: pioneer score 为启发式排序，empirical_p_value 来自 transition 置乱 null。"
        )

        path_for_neighbors = np.asarray(
            path_result.get("path_compute", path_result["path"]),
            dtype=float,
        )
        path_cell_indices = self._find_nearest_cells_along_path(path_for_neighbors, use_3d)
        gene_expression_along_path = self._get_gene_expression_along_path(path_cell_indices)
        gene_dynamics = self._analyze_gene_dynamics_near_transition(
            gene_expression_along_path, transition_idx, transition_window
        )

        all_pioneer_genes = self._identify_pioneer_genes(gene_dynamics, transition_idx)
        validated_genes = self.validate_pioneer_genes_with_biological_knowledge(
            all_pioneer_genes, transition_idx=transition_idx
        )
        if n_perm > 0 and validated_genes:
            validated_genes = self._add_permutation_pvalues(
                gene_expression_along_path,
                transition_idx,
                transition_window,
                validated_genes,
                n_perm=n_perm,
            )
        ranked_genes = self._rank_pioneer_genes(validated_genes, top_n_genes)

        return {
            'pioneer_genes': ranked_genes,
            'gene_expression_along_path': gene_expression_along_path,
            'path_cell_indices': path_cell_indices,
            'transition_idx': transition_idx,
            'gene_dynamics': gene_dynamics
        }

    def _find_nearest_cells_along_path(self, path, use_3d=False):
        """找到路径上每个点的最近邻细胞"""
        self.analyzer._ensure_3d_available(use_3d)
        cell_positions = (
            self.analyzer.cell_positions_3d
            if use_3d
            else self.analyzer.cell_positions_2d
        )

        nbrs = NearestNeighbors(n_neighbors=1).fit(cell_positions)
        distances, indices = nbrs.kneighbors(path)

        return indices.flatten()

    def _get_gene_expression_along_path(self, cell_indices):
        """提取沿路径的基因表达矩阵"""
        if sparse.issparse(self.adata.X):
            expression_matrix = self.adata.X[cell_indices, :].toarray()
        else:
            expression_matrix = self.adata.X[cell_indices, :]

        return expression_matrix

    @staticmethod
    def _load_tf_genes(paths):
        tf_genes = set()
        loaded_any = False
        for path in paths:
            try:
                series = pd.read_csv(path, header=None).iloc[:, 0].astype(str).str.strip().str.upper()
                tf_genes.update(g for g in series if g)
                loaded_any = True
            except FileNotFoundError:
                warnings.warn(
                    f"TF list not found: {path}; skip TF validation for this file.",
                    UserWarning,
                    stacklevel=2,
                )
        if not loaded_any:
            warnings.warn(
                "No TF gene lists loaded; TF validation disabled.",
                UserWarning,
                stacklevel=2,
            )
        return tf_genes

    def _compute_all_gene_means(self):
        x = self.adata.X
        if sparse.issparse(x):
            return np.asarray(x.mean(axis=0)).ravel()
        return np.mean(x, axis=0)

    def _smooth_expression(self, gene_expr):
        expr = np.asarray(gene_expr, dtype=float)
        window = self.smooth_window
        if window <= 1 or len(expr) < window:
            return expr
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(expr, kernel, mode="same")

    def _expression_level_score(self, mean_expression, gene_idx):
        if mean_expression < self.min_expression:
            return 0.0
        global_mean = float(self._all_gene_means[gene_idx])
        if global_mean <= 0:
            return min(1.0, mean_expression)
        return float(percentileofscore(self._all_gene_means, mean_expression, kind="rank") / 100.0)

    def _analyze_gene_dynamics_near_transition(self, expression_matrix, transition_idx, window_size):
        """
        分析基因在过渡态附近的表达动态

        根据文章中描述，先锋基因在过渡态附近：
        1. 表达水平发生剧烈变化
        2. 在过渡态达到稳定表达水平
        3. 变化时机与过渡态位置高度相关
        """
        n_genes = expression_matrix.shape[1]
        n_points = expression_matrix.shape[0]

        gene_dynamics = {}

        # 定义过渡态窗口
        window_start = max(0, transition_idx - window_size)
        window_end = min(n_points, transition_idx + window_size + 1)

        for gene_idx in range(n_genes):
            gene_expr = expression_matrix[:, gene_idx]
            gene_name = self.adata.var_names[gene_idx]

            dynamics_features = self._compute_gene_dynamics_features(
                gene_expr, transition_idx, window_start, window_end, gene_idx
            )

            gene_dynamics[gene_name] = dynamics_features

        return gene_dynamics

    def _compute_gene_dynamics_features(
        self, gene_expr, transition_idx, window_start, window_end, gene_idx
    ):
        """Compute path dynamics features on lightly smoothed expression."""
        gene_expr = self._smooth_expression(gene_expr)
        n_points = len(gene_expr)

        early_expr = gene_expr[:window_start]
        late_expr = gene_expr[window_start:transition_idx]

        if len(early_expr) >= 2 and len(late_expr) >= 2:
            early_response = np.mean(late_expr) - np.mean(early_expr)
            early_response_score = abs(early_response)
        else:
            early_response_score = 0.0

        expr_gradient = np.gradient(gene_expr)
        max_change_idx = int(np.argmax(np.abs(expr_gradient)))
        timing_proximity = 1.0 / (1.0 + abs(max_change_idx - transition_idx))

        pre_transition_mean = np.mean(gene_expr[: max(transition_idx, 1)])
        post_transition_mean = np.mean(gene_expr[transition_idx:]) if transition_idx < n_points else pre_transition_mean

        if pre_transition_mean > 0:
            fold_change = post_transition_mean / pre_transition_mean
            log2_fold_change = np.log2(fold_change) if fold_change > 0 else -10.0
        else:
            log2_fold_change = 10.0 if post_transition_mean > 0 else 0.0

        if transition_idx < n_points - 3:
            post_transition_trend = np.polyfit(
                range(3),
                gene_expr[transition_idx : transition_idx + 3],
                1,
            )[0]
            persistence_score = 1.0 / (1.0 + abs(post_transition_trend))
        else:
            persistence_score = 0.5

        if transition_idx < n_points - 5:
            post_stability_window = gene_expr[transition_idx:window_end]
            post_stability = 1.0 / (1.0 + np.std(post_stability_window))
        else:
            post_stability = 0.5

        second_derivative = np.gradient(np.gradient(gene_expr))
        smoothness = 1.0 / (1.0 + np.std(second_derivative))

        mean_expression = float(np.mean(gene_expr))
        expression_level_score = self._expression_level_score(mean_expression, gene_idx)

        early_up_pattern = self._detect_early_upregulation_pattern(gene_expr, transition_idx)
        early_down_pattern = self._detect_early_downregulation_pattern(gene_expr, transition_idx)
        pattern_score = max(early_up_pattern, early_down_pattern)

        expression_pattern = (
            "upregulated" if post_transition_mean > pre_transition_mean else "downregulated"
        )

        return {
            "early_response_score": early_response_score,
            "timing_proximity": timing_proximity,
            "log2_fold_change": abs(log2_fold_change),
            "persistence_score": persistence_score,
            "post_stability": post_stability,
            "smoothness": smoothness,
            "expression_level_score": expression_level_score,
            "pattern_score": pattern_score,
            "expression_pattern": expression_pattern,
            "pre_transition_mean": pre_transition_mean,
            "post_transition_mean": post_transition_mean,
            "max_change_idx": max_change_idx,
            "mean_expression": mean_expression,
            "transition_idx": transition_idx,
        }

    def _detect_early_upregulation_pattern(self, gene_expr, transition_idx):
        """检测早期上调模式：在过渡态前开始上升，之后稳定"""
        if transition_idx < 3:
            return 0

        # 检查过渡态前是否已经开始上升
        pre_trend = np.polyfit(range(3), gene_expr[transition_idx - 3:transition_idx], 1)[0]

        # 检查过渡态后是否稳定
        if len(gene_expr) > transition_idx + 2:
            post_trend = np.polyfit(range(3), gene_expr[transition_idx:transition_idx + 3], 1)[0]
            stability = 1.0 / (1.0 + abs(post_trend))
        else:
            stability = 0.5

        # 模式得分：早期上升 + 后期稳定
        pattern_score = (max(0, pre_trend) + stability) / 2
        return pattern_score

    def _detect_early_downregulation_pattern(self, gene_expr, transition_idx):
        """检测早期下调模式：在过渡态前开始下降，之后稳定"""
        if transition_idx < 3:
            return 0

        # 检查过渡态前是否已经开始下降
        pre_trend = np.polyfit(range(3), gene_expr[transition_idx - 3:transition_idx], 1)[0]

        # 检查过渡态后是否稳定
        if len(gene_expr) > transition_idx + 2:
            post_trend = np.polyfit(range(3), gene_expr[transition_idx:transition_idx + 3], 1)[0]
            stability = 1.0 / (1.0 + abs(post_trend))
        else:
            stability = 0.5

        # 模式得分：早期下降 + 后期稳定
        pattern_score = (max(0, -pre_trend) + stability) / 2
        return pattern_score

    def _score_from_features(self, features, transition_idx):
        """Heuristic composite score from dynamics features."""
        score = (
            0.15 * features["early_response_score"]
            + 0.15 * features["timing_proximity"]
            + 0.15 * min(5.0, features["log2_fold_change"]) / 5.0
            + 0.12 * features["persistence_score"]
            + 0.12 * features["post_stability"]
            + 0.10 * features["smoothness"]
            + 0.10 * features["expression_level_score"]
            + 0.11 * features["pattern_score"]
        )

        if features["log2_fold_change"] < 0.5:
            score *= 0.3
        if abs(features["max_change_idx"] - transition_idx) > 3:
            score *= 0.5
        if features["mean_expression"] < self.min_expression:
            score *= 0.4
        return float(score)

    def _identify_pioneer_genes(self, gene_dynamics, transition_idx):
        gene_scores = {}

        for gene_name, features in gene_dynamics.items():
            score = self._score_from_features(features, transition_idx)
            gene_scores[gene_name] = {
                "score": score,
                "features": features,
                "raw_score": score,
            }

        filtered_genes = {k: v for k, v in gene_scores.items() if v["score"] > 0.1}

        if not filtered_genes:
            print("警告: 没有基因满足先锋基因的标准")
            return {}

        return filtered_genes

    def validate_pioneer_genes_with_biological_knowledge(self, filtered_genes, transition_idx=None):
        validated_genes = {}

        for gene_name, gene_info in filtered_genes.items():
            validation_score = 1.0
            is_tf = self._is_transcription_factor(gene_name)
            if is_tf:
                validation_score *= 1.2

            pattern_validation = self._validate_expression_pattern(
                gene_info, transition_idx=transition_idx
            )
            validation_score *= pattern_validation

            validated_genes[gene_name] = {
                **gene_info,
                "validation_score": validation_score,
                "is_transcription_factor": is_tf,
                "score": gene_info["score"] * validation_score,
            }

        return validated_genes

    def _is_transcription_factor(self, gene_name):
        if not self.tf_genes:
            return False
        return gene_name.upper() in self.tf_genes

    def _add_permutation_pvalues(
        self, expression_matrix, transition_idx, window_size, validated_genes, n_perm
    ):
        """Empirical p-value by shuffling transition index along the path."""
        n_points = expression_matrix.shape[0]
        if n_points < 4:
            for info in validated_genes.values():
                info["empirical_p_value"] = np.nan
            return validated_genes

        rng = np.random.default_rng(42)
        gene_names = list(validated_genes.keys())
        gene_indices = [self.adata.var_names.get_loc(g) for g in gene_names]
        observed = np.array([validated_genes[g]["score"] for g in gene_names])
        exceed = np.zeros(len(gene_names), dtype=int)

        lo = max(1, window_size)
        hi = max(lo + 1, n_points - 1)

        for _ in range(n_perm):
            null_idx = int(rng.integers(lo, hi))
            wstart = max(0, null_idx - window_size)
            wend = min(n_points, null_idx + window_size + 1)
            for j, gene_idx in enumerate(gene_indices):
                expr = expression_matrix[:, gene_idx]
                feats = self._compute_gene_dynamics_features(
                    expr, null_idx, wstart, wend, gene_idx
                )
                null_score = self._score_from_features(feats, null_idx)
                if null_score >= observed[j]:
                    exceed[j] += 1

        for j, gene_name in enumerate(gene_names):
            validated_genes[gene_name]["empirical_p_value"] = (exceed[j] + 1) / (n_perm + 1)

        return validated_genes

    def _is_chromatin_related(self, gene_name):
        """检查基因是否与染色质相关"""
        chromatin_keywords = ['histone', 'h3', 'h4', 'h2a', 'h2b', 'chromatin', 'epigenetic']
        return any(keyword in gene_name.lower() for keyword in chromatin_keywords)

    def _validate_expression_pattern(self, gene_info, transition_idx=None):
        """Validate expression pattern using nested features dict."""
        features = gene_info.get("features", gene_info)
        score = 1.0
        t_idx = transition_idx
        if t_idx is None:
            t_idx = features.get("transition_idx", features.get("max_change_idx", 0))

        if features["max_change_idx"] > t_idx + 2:
            score *= 0.7
        if features["persistence_score"] < 0.5:
            score *= 0.8
        if features["mean_expression"] < self.min_expression:
            score *= 0.9
        return score

    def _rank_pioneer_genes(self, validated_genes, top_n_genes):
        """进一步排序和验证先锋基因"""

        # 按验证后得分重新排序
        reordered_genes = sorted(
            validated_genes.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        ranked_genes = {}
        for i, (gene_name, gene_info) in enumerate(reordered_genes[:top_n_genes]):
            ranked_genes[gene_name] = {'rank': i+1, **gene_info}

        return ranked_genes

    def _compute_expression_stability(self, gene_expr, transition_idx, stability_window=5):
        """计算基因表达稳定性得分"""

        # 过渡态后的表达稳定性
        post_transition_expr = gene_expr[transition_idx:]
        if len(post_transition_expr) > stability_window:
            post_stability = 1.0 / (1.0 + np.std(post_transition_expr[:stability_window]))
        else:
            post_stability = 1.0 / (1.0 + np.std(post_transition_expr))

        return post_stability

    def analyze_multiple_paths_pioneer_genes(self, all_path_results, top_n_genes=20, use_3d=False):
        """
        分析多个路径的先锋基因

        Parameters:
        -----------
        all_path_results : dict
            所有路径的计算结果
        top_n_genes : int
            每个路径返回的top N基因
        use_3d : bool
            是否使用3D路径

        Returns:
        --------
        all_pioneer_genes : dict
            所有路径的先锋基因分析结果
        """

        all_pioneer_genes = {}

        for path_key, path_result in all_path_results.items():
            print(f"分析路径 {path_key} 的先锋基因...")

            pioneer_result = self.identify_pioneer_genes_along_path(
                path_result,
                top_n_genes=top_n_genes,
                use_3d=use_3d
            )

            all_pioneer_genes[path_key] = pioneer_result

        return all_pioneer_genes

    def plot_gene_expression_dynamics(self, pioneer_result, top_genes=20, figsize=(12, 10), save_path=None):
        """
        绘制先锋基因沿路径的表达动态

        Parameters:
        -----------
        pioneer_result : dict
            先锋基因分析结果
        top_genes : int
            显示top多少基因
        figsize : tuple
            图像大小
        save_path : str
            保存路径
        """

        gene_expression = pioneer_result['gene_expression_along_path']
        path_cell_indices = pioneer_result['path_cell_indices']
        transition_idx = pioneer_result['transition_idx']
        pioneer_genes = pioneer_result['pioneer_genes']

        # 选择top基因
        top_gene_names = list(pioneer_genes.keys())[:top_genes]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axes.flatten()

        # 1. 绘制所有top基因的表达热图
        self._plot_gene_expression_heatmap(ax1, gene_expression, top_gene_names, transition_idx)

        # 2. 绘制代表性基因的表达轨迹
        self._plot_representative_gene_trajectories(ax2, gene_expression, top_gene_names, transition_idx)

        # 3. 绘制基因得分分布
        self._plot_gene_scores(ax3, pioneer_genes, top_genes)

        # 4. 绘制过渡态附近的基因表达变化
        self._plot_transition_expression_changes(ax4, pioneer_genes, gene_expression, transition_idx)

        plt.tight_layout()
        plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.16, bottom=0.1)  # 为图例留出空

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

        return fig, axes

    def _plot_gene_expression_heatmap(self, ax, gene_expression, gene_names, transition_idx):
        """绘制基因表达热图"""

        # 获取基因索引
        gene_indices = [self.adata.var_names.get_loc(gene) for gene in gene_names]
        expression_data = gene_expression[:, gene_indices]

        # 标准化表达数据
        try:
            z = zscore(expression_data, axis=0, nan_policy="omit")
        except TypeError:
            z = zscore(expression_data, axis=0)
        expression_data_zscore = np.nan_to_num(z)

        im = ax.imshow(expression_data_zscore.T, aspect='auto', cmap='viridis', interpolation='nearest')

        # 标记过渡态
        ax.axvline(x=transition_idx, color='red', linestyle='--', linewidth=2, label='Transition State')

        ax.set_yticks(range(len(gene_names)))
        ax.set_yticklabels(gene_names, fontsize=8)
        ax.set_xlabel('Path Position', fontsize=9)
        ax.set_title('Gene Expression Dynamics Along Path (Z-score)', fontsize=10)
        #ax.legend()

        plt.colorbar(im, ax=ax, label='Z-score')

    def _plot_representative_gene_trajectories(self, ax, gene_expression, gene_names, transition_idx):
        """绘制代表性基因的表达轨迹"""

        # 选择前几个基因进行绘制
        plot_genes = gene_names[:min(10, len(gene_names))]

        for i, gene_name in enumerate(plot_genes):
            gene_idx = self.adata.var_names.get_loc(gene_name)
            expr = gene_expression[:, gene_idx]

            # 标准化表达
            denom = np.max(expr) - np.min(expr)
            expr_normalized = (expr - np.min(expr)) / (denom + 1e-8)

            ax.plot(expr_normalized, label=gene_name, linewidth=1)

        #ax.axvline(x=transition_idx, color='red', linestyle='--',linewidth=1.5, label='Transition State')
        ax.axvline(x=transition_idx, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Path Position', fontsize=9)
        ax.set_ylabel('Normalized Expression', fontsize=9)
        ax.set_title('Representative Gene Expression Trajectories', fontsize=10)
        #ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)

    def _plot_gene_scores(self, ax, pioneer_genes, top_genes):
        """绘制基因得分分布"""

        gene_names = list(pioneer_genes.keys())[:top_genes]
        scores = [pioneer_genes[gene]["score"] for gene in gene_names]

        colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))

        ax.barh(range(len(scores)), scores, color=colors)
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels(gene_names, fontsize=8)
        ax.set_xlabel("Heuristic remodeling score", fontsize=9)
        ax.set_title("Remodeling-associated candidate genes (composite score)", fontsize=10)

    def _plot_transition_expression_changes(self, ax, pioneer_genes, gene_expression, transition_idx):
        """绘制过渡态附近的基因表达变化"""

        # 计算每个基因在过渡态前后的表达变化
        changes = []
        gene_names = list(pioneer_genes.keys())

        for gene_name in gene_names:
            gene_idx = self.adata.var_names.get_loc(gene_name)
            expr = gene_expression[:, gene_idx]

            pre_mean = np.mean(expr[:transition_idx])
            post_mean = np.mean(expr[transition_idx:])
            change = post_mean - pre_mean

            changes.append(change)

        # 绘制变化分布
        colors = ['red' if change < 0 else 'blue' for change in changes]
        bars = ax.bar(range(len(changes)), changes, color=colors, alpha=0.7)

        ax.set_xticks(range(len(changes)))
        ax.set_xticklabels(gene_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Expression Change (Post - Pre Transition)', fontsize=9)
        ax.set_title('Gene Expression Changes Across Transition', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Upregulated'),
            Patch(facecolor='red', label='Downregulated')
        ]
        ax.legend(handles=legend_elements, fontsize=8)


# 整合到主分析流程中的函数
def analyze_cell_fate_with_pioneer_genes(adata, start_state, end_state,
                                         clustering_key='stage',
                                         use_3d=False, n_path_points=100,
                                         top_n_genes=15):
    """
    完整的细胞命运分析流程，包括先锋基因识别
    """

    print("=== 细胞命运景观与先锋基因分析 ===")

    # 初始化景观分析器
    analyzer = NonEquilibriumCellFateLandscape(
        adata,
        potential_key='potential',
        embedding_2d_key='X_umap',
        potential_transform="none",
    )

    # 识别细胞状态
    cell_states = analyzer.identify_cell_states(clustering_key=clustering_key, use_3d=use_3d)

    if start_state not in cell_states:
        raise ValueError(f"起始状态 {start_state} 未找到")
    if end_state not in cell_states:
        raise ValueError(f"结束状态 {end_state} 未找到")

    start_pos = cell_states[start_state]['position']
    end_pos = cell_states[end_state]['position']

    # 计算最小作用路径
    print("计算最小作用路径...")
    path_result = analyzer.compute_least_action_path(start_pos, end_pos, n_points=n_path_points, use_3d=use_3d)

    # 初始化先锋基因识别器
    pioneer_identifier = PioneerGeneIdentifier(analyzer)

    # 识别先锋基因
    print("识别先锋基因...")
    pioneer_result = pioneer_identifier.identify_pioneer_genes_along_path(path_result, top_n_genes=top_n_genes, use_3d=use_3d)

    # 打印结果
    print(f"\n=== 先锋基因分析结果 (heuristic ranking): {start_state} -> {end_state} ===")
    for i, (gene_name, gene_info) in enumerate(pioneer_result['pioneer_genes'].items()):
        pval = gene_info.get("empirical_p_value", np.nan)
        pval_str = f"{pval:.3f}" if np.isfinite(pval) else "n/a"
        print(
            f"{i + 1:2d}. {gene_name:15s} | Score: {gene_info['score']:.4f} | "
            f"Validation: {gene_info['validation_score']:.2f} | p_perm: {pval_str}"
        )
    visualizer = LandscapeVisualizer(analyzer)

    # 绘制景观和路径
    fig1, ax1 = visualizer.plot_landscape_with_path(path_result, use_3d=use_3d, show_flux=True)

    # 绘制基因表达动态
    fig2, axes2 = pioneer_identifier.plot_gene_expression_dynamics(pioneer_result)

    return {
        'analyzer': analyzer,
        'path_result': path_result,
        'pioneer_identifier': pioneer_identifier,
        'pioneer_result': pioneer_result,
        'cell_states': cell_states,
        'visualizer': visualizer
    }


def analyze_multiple_paths_with_pioneer_genes(adata, path_pairs,
                                              clustering_key='stage',
                                              use_3d=False, n_path_points=100,
                                              top_n_genes=15):
    """
    多路径分析，包括先锋基因识别
    """

    print("=== 多路径细胞命运与先锋基因分析 ===")

    # 初始化景观分析器
    analyzer = NonEquilibriumCellFateLandscape(
        adata,
        potential_key='potential',
        embedding_2d_key='X_umap',
        potential_transform="none",
    )

    # 识别细胞状态
    cell_states = analyzer.identify_cell_states(
        clustering_key=clustering_key,
        use_3d=use_3d
    )

    # 验证所有状态对
    valid_pairs = []
    for start_state, end_state in path_pairs:
        if start_state in cell_states and end_state in cell_states:
            valid_pairs.append((start_state, end_state))
        else:
            print(f"警告: 状态对 {start_state}->{end_state} 无效，跳过")

    # 计算所有路径
    all_path_results = analyzer.compute_multiple_paths(
        valid_pairs,
        use_3d=use_3d,
        n_points=n_path_points
    )

    # 初始化先锋基因识别器
    pioneer_identifier = PioneerGeneIdentifier(analyzer)

    # 识别所有路径的先锋基因
    all_pioneer_genes = pioneer_identifier.analyze_multiple_paths_pioneer_genes(
        all_path_results,
        top_n_genes=top_n_genes,
        use_3d=use_3d
    )

    # 打印综合结果
    print(f"\n=== 多路径先锋基因综合分析 ===")

    # 找出在所有路径中频繁出现的基因（核心调控基因）
    common_genes = {}
    for path_key, pioneer_result in all_pioneer_genes.items():
        for gene_name in pioneer_result['pioneer_genes'].keys():
            if gene_name not in common_genes:
                common_genes[gene_name] = []
            common_genes[gene_name].append(path_key)

    # 按出现频率排序
    common_genes_sorted = sorted(common_genes.items(), key=lambda x: len(x[1]), reverse=True)

    print("频繁出现的核心调控基因:")
    for gene_name, paths in common_genes_sorted[:10]:
        print(f"  {gene_name}: 出现在 {len(paths)} 个路径中 - {paths}")

    # 可视化
    visualizer = LandscapeVisualizer(analyzer)

    # 绘制多路径景观
    fig1, ax1 = visualizer.plot_landscape_with_multiple_paths(
        all_path_results,
        use_3d=use_3d,
        show_flux=True
    )

    # 为每个路径绘制基因表达动态
    for path_key, pioneer_result in all_pioneer_genes.items():
        print(f"\n绘制路径 {path_key} 的基因表达动态...")
        fig, axes = pioneer_identifier.plot_gene_expression_dynamics(pioneer_result)
        fig.suptitle(f"Gene Expression Dynamics - {path_key}", fontsize=16)

    return {
        'analyzer': analyzer,
        'all_path_results': all_path_results,
        'pioneer_identifier': pioneer_identifier,
        'all_pioneer_genes': all_pioneer_genes,
        'common_genes': common_genes_sorted,
        'cell_states': cell_states,
        'visualizer': visualizer
    }



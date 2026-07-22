# v1.3 Paper Repositioning Initial Revision

这版修改的核心目标：把论文创新点从容易被质疑的“启发式 + RL 结合”改成更贴合项目实现的“后决策/反事实前瞻监督信号蒸馏 + 在线动作级候选排序”。KSP 仍然是路径骨架来源，因此论文中不应声称“异质路径生成”是主要创新；更准确的说法是：在 KSP 路径骨架上构造异质的合法 RMSA 动作候选，并用学习到的 ranker 做最终选择。

## 1. 建议标题

原题：

```latex
\title{Hybrid Candidate Generation and Learned Ranking for Compute-Optical Co-Design in Metro Optical Networks}
```

建议改为：

```latex
\title{Post-Decision-Guided Action Ranking for Compute-Optical Co-Design in Metro Optical Networks}
```

备选：

```latex
\title{Amortized Lookahead Action Ranking for Compute-Optical Co-Design in Metro Optical Networks}
```

更稳妥的长标题：

```latex
\title{Post-Decision-Guided RMSA Action Ranking for Distributed DNN Inference over Compute-Optical Networks}
```

## 2. 摘要替换稿

```latex
\begin{abstract}
Compute-optical co-design couples DNN partition/offloading decisions with elastic optical routing, modulation, and spectrum assignment (RMSA). In this setting, the computation-side (C-side) decision determines the split point and target MEC server, while the routing-side (R-side) decision must allocate a path, modulation format, and contiguous spectrum block under deadline, compute-capacity, and spectrum-continuity constraints. This closed-loop structure makes the value of an RMSA action strongly state-dependent: an action that is locally feasible may still create unfavorable future fragmentation or interact poorly with the C-side load distribution.

This paper proposes SA-HMARL, a hierarchical compute-optical decision framework with post-decision-guided R-side action ranking. The C-side uses a masked PPO policy to select a joint split-server action. Conditioned on this decision, the R-side first enumerates legal path-modulation-spectrum-block actions over a K-shortest-path backbone, then retrieves a compact action-level candidate set from PPO-R proposals, spectrum-aware anchors, shortest-path and KSP-FF anchors, and high-logit legal fillers. Instead of using PPO-R as the final controller, SA-HMARL trains a learned ranker from offline post-decision and H-step counterfactual labels. The expensive lookahead evaluation is performed only offline; online inference only scores a bounded candidate set and selects the highest-ranked legal RMSA action.

Long-horizon evaluations show that the proposed ranker can reduce blocking against strong KSP-FF and DeepRMSA-style baselines when the C-side policy preserves a useful feasible action region. Ablation studies further show that R-side ranking gains are conditional on adaptive split-server selection: when the C-side collapses the feasible region through weak or fixed split decisions, little recoverable R-side headroom remains. These results suggest that the main benefit of SA-HMARL lies in amortizing post-decision lookahead into an efficient online RMSA ranker, rather than merely combining heuristic candidate generation with reinforcement learning.
\end{abstract}
```

## 3. 引言核心叙事替换稿

建议替换掉原文中“heuristic and learned methods are complementary / hybrid candidate generation is the main idea”那几段，改成下面这版：

```latex
Deep reinforcement learning has been widely studied for RMSA because it can learn state-dependent decisions under dynamic traffic. However, directly using a policy-gradient actor as the final R-side controller has two limitations in our compute-optical setting. First, the R-side action is not merely a path or a spectrum index; it is a legal path-modulation-spectrum-block tuple whose feasibility depends on the C-side split-server decision, deadline slack, modulation reach, and current spectrum occupancy. Second, a one-step policy target does not explicitly evaluate how an admitted action changes the post-decision resource state, e.g., whether it preserves future spectrum flexibility or creates fragmentation that increases future blocking.

The key design question is therefore not whether RMSA should be solved by heuristics or learning alone. K-shortest-path search remains a practical and widely used backbone for exposing feasible optical routes, and DeepRMSA-style methods also rely on KSP-generated candidate paths. The more important question is how to evaluate the legal RMSA actions induced by this backbone under the current compute-optical state. In particular, the controller should account for the post-decision state created by an action and the future requests that may arrive after the current admission decision.

Motivated by this observation, we reformulate R-side control as an action-ranking problem guided by offline post-decision lookahead. SA-HMARL first uses KSP to construct the path backbone and enumerates legal path-modulation-block actions under the selected C-side split-server pair. It then forms a bounded online candidate pool by mixing PPO-R proposals, spectrum-aware action anchors, a shortest-path anchor, an explicit KSP-FF anchor, and high-logit legal fillers. The final action is not selected by PPO-R or by a fixed first-fit rule. Instead, a learned ranker scores candidate actions using C-side context, optical features, and action-level structural features. The ranker is trained from offline labels obtained by executing each candidate in a copied environment and estimating its post-decision quality through finite-horizon counterfactual rollouts.

This design turns expensive H-step lookahead into an amortized online model. During training, the system may evaluate many candidate actions by copying the environment, applying each action, and rolling out future requests under frozen policies. During deployment, however, the R-side only enumerates legal actions, retrieves a small candidate subset, computes feature vectors, and performs one ranker forward pass. Thus, the online decision cost remains in the same order as KSP-FF and DeepRMSA-style candidate evaluation, with a controlled constant overhead from ranking.
```

## 4. 贡献列表替换稿

```latex
The main contributions of this work are as follows:

\begin{enumerate}
\item We formulate closed-loop compute-optical co-design for distributed DNN inference, where the controller jointly considers C-side split-server selection and R-side path-modulation-spectrum-block assignment under compute-capacity, deadline, modulation-reach, and spectrum-continuity constraints.

\item We identify the R-side bottleneck as a post-decision action-valuation problem rather than a purely path-generation problem. Since both KSP-FF and DeepRMSA-style methods rely on KSP-generated path backbones, the key challenge is to rank legal RMSA actions according to their current and future resource impact.

\item We propose a post-decision-guided R-side ranker. Offline, candidate RMSA actions are evaluated through copied-environment execution and finite-horizon counterfactual rollouts. Online, the learned ranker amortizes this expensive lookahead by scoring a bounded action-level candidate set.

\item We design a practical online candidate-retrieval mechanism over the KSP backbone. The candidate set contains PPO-R proposals, per-path spectrum anchors, a shortest-path anchor, high-logit legal fillers, and an explicit KSP-FF anchor, ensuring that the strong heuristic action remains available while allowing the ranker to select better non-greedy actions when predicted to be beneficial.

\item We provide long-horizon cross-topology evaluations and C-side ablations showing that R-side ranking gains depend on the feasible region created by adaptive split-server selection. This clarifies the division of labor between PPO-C, which shapes compute and optical feasibility, and the R-side ranker, which improves RMSA action quality within that feasible region.
\end{enumerate}
```

## 5. 架构章节替换稿

建议把 Section IV 开头改成如下口径：

```latex
\section{SA-HMARL Architecture}
\label{sec:architecture}

SA-HMARL follows a hierarchical request-level decision structure. At each request arrival, expired services are first released and the current compute-optical state $\Omega_t$ is updated. The controller then makes a C-side decision followed by an R-side decision:
\begin{equation}
r_t,\Omega_t
\rightarrow o_C
\rightarrow a_C=(s,v)
\rightarrow o_R(s,v)
\rightarrow \mathcal{A}_{\mathrm{cand}}
\rightarrow a_R^*
\rightarrow \Omega_{t+1}.
\label{eq:sahmarl_pipeline}
\end{equation}
The C-side action $a_C=(s,v)$ selects one DNN split point and one target MEC server. Conditioned on this selected split-server pair, the R-side action $a_R=(p,m,b)$ selects one optical path, one modulation format, and one contiguous spectrum block.

This hierarchy is not meant to claim that path generation itself is learned or novel. The path backbone is generated by K-shortest-path search, as in many RMSA systems. The key distinction of SA-HMARL is that the final R-side controller is not a one-step PPO actor and not a static KSP-FF rule. Instead, it is an amortized ranker trained from post-decision and finite-horizon counterfactual labels. The ranker evaluates legal action tuples under the current C-side context and optical resource state.
```

### C-side subsection

```latex
\subsection{C-side Masked PPO for Split-Server Selection}
\label{subsec:c_side_policy}

The C-side action space is the Cartesian product of split candidates and MEC servers,
\begin{equation}
\mathcal{A}_C(r)=\mathcal{S}_r\times V^{\mathrm{mec}}.
\label{eq:c_action_space}
\end{equation}
PPO-C directly selects a joint split-server action rather than choosing the split and server independently. For each candidate $(s,v)$, the observation includes split-profile features, server residual capacity and utilization, estimated local and edge compute delays, path-distance summaries, and a coarse downstream optical feasibility count. A feasibility mask removes actions whose edge compute demand exceeds the residual server capacity or whose downstream R-side feasible-action count is zero.

During deterministic evaluation, PPO-C selects
\begin{equation}
a_C^*=(s^*,v^*)=
\arg\max_{a_C\in\mathcal{A}_C^{\mathrm{legal}}}
\pi_C(a_C|o_C).
\label{eq:c_policy_select}
\end{equation}
This action fixes the intermediate feature size, edge compute demand, destination server, and source-destination pair for the subsequent RMSA decision. Therefore, PPO-C shapes the feasible region in which the R-side ranker operates.
```

### R-side legal action enumeration subsection

```latex
\subsection{R-side Legal Action Enumeration over a KSP Backbone}
\label{subsec:r_legal_enum}

Given $(s^*,v^*)$, the R-side first obtains a K-shortest-path candidate set
\begin{equation}
\mathcal{P}_{r,v^*}^{K}=\{p_1,p_2,\ldots,p_K\}.
\label{eq:ksp_backbone}
\end{equation}
For every path $p\in\mathcal{P}_{r,v^*}^{K}$ and modulation format $m\in\mathcal{M}$, the controller checks modulation reach and computes the required number of frequency slots from the remaining deadline budget. It then extracts contiguous idle spectrum blocks from the intersection of idle-slot vectors over all links on $p$.

The legal R-side action set is
\begin{equation}
\mathcal{A}_R^{\mathrm{legal}}
=
\{(p,m,b):p\in\mathcal{P}_{r,v^*}^{K},m\in\mathcal{M},
b\in\mathcal{B}_{r,s^*,v^*,p,m}(t),\chi(p,m,b)=1\},
\label{eq:r_legal_actions}
\end{equation}
where $\chi(p,m,b)$ indicates that the action satisfies modulation reach, positive transmission-time budget, sufficient block size, spectrum continuity, contiguity, and non-overlap with active services. If $\mathcal{A}_R^{\mathrm{legal}}$ is empty, the request is blocked under the selected C-side action.
```

### R-side candidate retrieval subsection

```latex
\subsection{Bounded Action-Level Candidate Retrieval}
\label{subsec:r_candidate_retrieval}

SA-HMARL does not rank all legal actions when the legal set is large. Instead, it constructs a bounded action-level candidate set
\begin{equation}
\mathcal{A}_{\mathrm{cand}}\subseteq \mathcal{A}_R^{\mathrm{legal}},
\qquad
|\mathcal{A}_{\mathrm{cand}}|\le K_{\max}.
\label{eq:bounded_candidates}
\end{equation}
The default online configuration uses $K_{\max}=48$ and explicitly preserves the current KSP-FF action when it is legal.

The candidate set is assembled as
\begin{equation}
\mathcal{A}_{\mathrm{cand}}
=
\operatorname{Trunc}_{K_{\max}}
\left(
\mathcal{A}_{\mathrm{ppo}}
\cup
\mathcal{A}_{\mathrm{path}}
\cup
\mathcal{A}_{\mathrm{short}}
\cup
\mathcal{A}_{\mathrm{fill}}
\cup
\mathcal{A}_{\mathrm{ksp}}
\right).
\label{eq:candidate_union}
\end{equation}
Here $\mathcal{A}_{\mathrm{ppo}}$ contains top-logit legal actions proposed by PPO-R. PPO-R is used only as a proposer, not as the deployed final controller. The per-path anchor set $\mathcal{A}_{\mathrm{path}}$ adds, for each path, representative legal actions such as the first available block, closest-fit block, largest block, and minimum-waste block. The set $\mathcal{A}_{\mathrm{short}}$ contains a shortest-path legal action. The filler set $\mathcal{A}_{\mathrm{fill}}$ uses high-logit remaining legal actions to fill unused candidate slots. Finally, $\mathcal{A}_{\mathrm{ksp}}$ injects the KSP-FF action if it is legal:
\begin{equation}
\mathcal{A}_{\mathrm{ksp}}=
\begin{cases}
\{a_{\mathrm{ksp}}\}, & a_{\mathrm{ksp}}\in\mathcal{A}_R^{\mathrm{legal}},\\
\emptyset, & \text{otherwise}.
\end{cases}
\label{eq:ksp_ff_anchor}
\end{equation}

This design should be interpreted as heterogeneous action retrieval over a KSP path backbone. The paths themselves are still generated by KSP; diversity comes from the action-level combination of path, modulation, spectrum block, PPO proposal, spectrum-aware anchors, and the preserved KSP-FF baseline action.
```

### Ranker subsection

```latex
\subsection{Post-Decision-Guided Action Ranking}
\label{subsec:post_decision_ranker}

For each candidate action $a_R\in\mathcal{A}_{\mathrm{cand}}$, SA-HMARL builds a state-action feature vector
\begin{equation}
\phi_R(o_C,o_R,a_R),
\label{eq:r_ranker_feature}
\end{equation}
including path length, hop count, largest free block, free-slot ratio, fragmentation index, modulation spectral efficiency and reach, required FS count, candidate block size, block waste, selected split, selected server, deadline, holding time, intermediate data size, server utilization, raw R-side valid-action ratio, and normalized action indices.

The learned ranker assigns a scalar score
\begin{equation}
q_\theta(o_C,o_R,a_R)=f_\theta(\phi_R(o_C,o_R,a_R)).
\label{eq:ranker_score}
\end{equation}
The final R-side action is
\begin{equation}
a_R^*=
\arg\max_{a_R\in\mathcal{A}_{\mathrm{cand}}}
q_\theta(o_C,o_R,a_R).
\label{eq:ranker_select}
\end{equation}
The ranker is therefore the deployed finalizer. It can select the KSP-FF action when that action receives the highest score, but it can also choose a non-first-fit action when the learned post-decision value predicts better downstream behavior.
```

## 6. 在线算法替换稿

```latex
\begin{algorithm}[t]
\caption{Online SA-HMARL Decision Procedure}
\label{alg:online_sahmarl}
\begin{algorithmic}[1]
\REQUIRE Request $r_t$, current state $\Omega_t$, PPO-C policy $\pi_C$, PPO-R proposer $\pi_R$, ranker $f_\theta$, candidate budget $K_{\max}$
\STATE Release expired services and update server loads and spectrum occupancy.
\STATE Build C-side observation $o_C$ and legal C-side mask.
\IF{no legal C-side action exists}
    \STATE Block $r_t$ and return.
\ENDIF
\STATE Select $a_C^*=(s^*,v^*)$ using masked PPO-C.
\STATE Build R-side observation $o_R(s^*,v^*)$.
\STATE Generate KSP path backbone $\mathcal{P}_{r,v^*}^{K}$.
\STATE Enumerate legal R-side actions $\mathcal{A}_R^{\mathrm{legal}}$ over path, modulation, and spectrum-block tuples.
\IF{$\mathcal{A}_R^{\mathrm{legal}}=\emptyset$}
    \STATE Block $r_t$ and return.
\ENDIF
\STATE Build $\mathcal{A}_{\mathrm{cand}}$ from PPO-R proposals, per-path spectrum anchors, shortest-path anchor, high-logit fillers, and legal KSP-FF anchor.
\STATE Build $\phi_R(o_C,o_R,a_R)$ for each $a_R\in\mathcal{A}_{\mathrm{cand}}$.
\STATE Score each candidate with $q_\theta=f_\theta(\phi_R)$.
\STATE Select $a_R^*=\arg\max_{a_R\in\mathcal{A}_{\mathrm{cand}}} q_\theta(o_C,o_R,a_R)$.
\STATE Execute $(a_C^*,a_R^*)$; reserve server compute and optical spectrum until the holding time expires.
\end{algorithmic}
\end{algorithm}
```

## 7. Training and Candidate-Ranking Methodology 替换稿

```latex
\section{Training and Candidate-Ranking Methodology}
\label{sec:training}

\subsection{Overview}
\label{subsec:training_overview}

SA-HMARL separates online control from offline action valuation. PPO-C and PPO-R are first trained as masked policies using standard rollout interaction with the simulator. PPO-C is retained as the deployed C-side controller. PPO-R is not used as the final deployed R-side controller in v1.3; instead, it provides proposal logits for candidate retrieval and continuation behavior during counterfactual data generation.

The R-side ranker is trained offline. For sampled decision states, the generator freezes the C-side policy, selects the same split-server action as would be used online, constructs the corresponding legal R-side action set, and evaluates candidate R actions by copying the environment. Each candidate is applied to an identical pre-R-decision snapshot, followed by a finite-horizon rollout under frozen policies on the same future request trace. This produces comparable labels for actions within the same decision state.

\subsection{Post-Decision and Counterfactual Labels}
\label{subsec:counterfactual_labels}

Let $\Omega_t$ be the state immediately before the R-side decision and let $a_C^*=(s^*,v^*)$ be the selected C-side action. For a candidate R-side action $a_R$, define the post-decision state
\begin{equation}
\Omega_t^+(a_R)=T(\Omega_t,a_C^*,a_R),
\label{eq:post_decision_state}
\end{equation}
where $T(\cdot)$ denotes the simulator transition that reserves compute and spectrum resources if the action is feasible. The offline evaluator then rolls out $H$ future requests from $\Omega_t^+(a_R)$ under frozen continuation policies and obtains a finite-horizon estimate
\begin{equation}
G_H(a_R)
=
\sum_{\tau=0}^{H}
\gamma^\tau r_{t+\tau}.
\label{eq:h_step_return}
\end{equation}
In practice, the label also includes spectrum-health and delay-related terms, such as post-decision spectrum flexibility, current blocking, future blocking, future no-suitable-block events, and normalized delay. A generic label can be written as
\begin{equation}
y(a_R)
=
\alpha_\phi \Phi(\Omega_t^+(a_R))
-\alpha_0 B_t(a_R)
-\alpha_H B_{t:t+H}(a_R)
-\alpha_N N_{t:t+H}(a_R)
-\alpha_D \bar{D}(a_R),
\label{eq:ranker_label}
\end{equation}
where $\Phi$ measures post-decision spectrum health, $B_t$ is current blocking, $B_{t:t+H}$ is future blocking count, $N_{t:t+H}$ is future no-suitable-block count, and $\bar{D}$ is normalized delay.

For each sampled state, labels are stored as a ranked group rather than as independent samples. This is important because the absolute label scale can vary across traffic states, while the rank order among candidate actions in the same state directly expresses which RMSA action should be preferred.

\subsection{Listwise Ranker Training}
\label{subsec:listwise_training}

Given a group of candidate actions $\mathcal{A}_{\mathrm{cand}}(t)$ and labels $\{y(a_R)\}$, the ranker is trained to assign higher scores to actions with better post-decision and finite-horizon outcomes. The training objective can be implemented using a supervised regression or listwise ranking loss:
\begin{equation}
\mathcal{L}(\theta)
=
\sum_t
\ell_{\mathrm{rank}}
\left(
\{f_\theta(\phi_R(o_C,o_R,a_R))\}_{a_R\in\mathcal{A}_{\mathrm{cand}}(t)},
\{y(a_R)\}_{a_R\in\mathcal{A}_{\mathrm{cand}}(t)}
\right).
\label{eq:ranker_loss}
\end{equation}
The trained model therefore distills expensive copied-environment H-step evaluation into a cheap online scoring function.

\subsection{Online Complexity}
\label{subsec:online_complexity}

The H-step counterfactual evaluation is deliberately offline and can be expensive because it branches over candidate actions and rolls out future requests. This cost is not paid at deployment time. Online, SA-HMARL performs four operations: legal action enumeration over the KSP backbone, bounded candidate retrieval, feature construction, and one ranker forward pass over at most $K_{\max}$ candidates.

If $K$ is the number of candidate paths, $|\mathcal{M}|$ is the number of modulation formats, $N_f$ is the number of spectrum slots, and $K_{\max}$ is the ranker candidate budget, the online R-side cost is dominated by legal-action construction and bounded ranking:
\begin{equation}
O(K|\mathcal{M}|N_f) + O(K_{\max}d_f) + O(K_{\max}C_\theta),
\label{eq:online_ranker_complexity}
\end{equation}
where $d_f$ is the feature dimension and $C_\theta$ is the per-candidate ranker inference cost. With a fixed $K_{\max}$, the online latency remains in the same asymptotic order as KSP-based candidate evaluation, with a controlled constant overhead for neural scoring.
```

## 8. 必须删改的旧口径

下面这些表达建议从全文删掉或改弱：

- 不要说“v1.3 的创新是启发式和 RL 结合”。DeepRMSA 本身也使用 KSP 候选路径和 RL 动作选择，这样写容易被审稿人反驳。
- 不要说“候选路径具有异质性”。更准确是“候选 RMSA 动作具有异质性”，路径骨架仍来自 KSP。
- 不要把 PPO-R 写成最终 R-side controller。v1.3 中 PPO-R 是 proposer，ranker 才是 finalizer。
- 不要说在线阶段执行 H-step 前瞻。H-step/counterfactual rollout 是离线数据生成，在线只是 bounded candidate ranking。
- 不要宣称 v1.3 在所有拓扑、所有 C-side 策略下都稳定优于 KSP-FF。更稳的是说：在 learned PPO-C 能保持有效 feasible region 时，R-side ranker 有收益；在 fixed/weak C-side 下，收益会明显收缩。

## 9. 推荐论文主线一句话

可以在引言或结论中反复使用这句话：

```latex
The proposed R-side method is best understood as amortized post-decision lookahead: expensive counterfactual evaluation is used offline to train an action ranker, while online control reduces to bounded legal-candidate scoring over a KSP path backbone.
```


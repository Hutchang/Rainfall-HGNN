arXiv:2607.01621v1 [cs.AI] 2 Jul 2026

# Spatial Support Matters: Geometry-Aware Graph Fusion for Rainfall Field Reconstruction

**Low Jun Yu<sup>1,∗</sup> Niramay Kachhadiya<sup>2</sup> Herath Mudiyanselage Viraj Vidura Herath<sup>2,+</sup>**
**Sanka Rasnayaka<sup>1</sup> and Lucy Amanda Marshall<sup>2</sup>**

<sup>1</sup>School of Computing, National University of Singapore, <sup>2</sup>Faculty of Engineering, The University of Sydney

<sup>∗</sup>lowjy28@gmail.com, <sup>+</sup>viraj.herath@sydney.edu.au

### Abstract

*Fine-scale rainfall reconstruction is critical for urban flood modeling, but real rainfall sensing systems observe the field through incompatible spatial supports: gauges measure points, microwave links measure paths, and radar/satellite products measure gridded areas. These differences in measurement support impose geometrically distinct constraints on the rainfall field, yet existing heterogeneous graph approaches reconcile such sources in feature space, giving each its own embedding while discarding the geometry of its support. We propose a geometry-aware multi-support heterogeneous graph neural network that represents each observation according to its support type (0D point, 1D line, or 2D grid) as a distinct node layer, and fuses them through cross-support message passing into a point-support prediction layer from which the field is reconstructed. An inductive masked-node formulation decouples prediction resolution from sensing resolution, allowing the same trained model to reconstruct the field at user-defined target locations or display grids. On Singapore data, the proposed method reduces RMSE by 23.2% over the classical interpolation baseline, inverse-distance weighting, and consistently outperforms other neural architectures such as convolutional fusion and support-agnostic heterogeneous graph baselines. A generalization study using data from Sydney, Australia lets us characterize when multi-support fusion helps: the available skill appears to depend on gauge spacing relative to the spatial correlation length of the field, so fusion delivers the largest gains where the field is under-sampled relative to its correlation length and little when it is already resolved. Code and models are open-sourced at \<ANONYMIZED\>.*

## 1. Introduction

Fine-scale rainfall reconstruction is critical for urban flood modeling, water management, and disaster response, yet 

 rainfall remains one of the hardest geophysical fields to measure [1]. Rainfall is spatially localized, intermittent, and rapidly evolving, with convective storms producing sharp gradients over a few kilometers that no single instrument captures well. Rain gauges sample it accurately but only at isolated points; weather radar and satellites cover wide areas but observe the field indirectly and aloft; commercial microwave links (CMLs) measure it along the paths between telecommunication towers. Each instrument senses the same field through a different spatial geometry, and no single source is sufficient alone.

The difficulty is not only that the sensors are heterogeneous, but that they observe rainfall over different spatial supports: gauges measure point values, microwave links measure path-integrated attenuation, and radar or satellite products estimate areal averages over grid cells. We refer to this geometric measurement region as the *spatial support* of an observation. These differences in support are not identical: a point measurement, a path integral, and an areal average impose geometrically distinct constraints on the field, and fusing them well requires respecting that geometry.

Classical interpolation techniques such as inverse-distance weighting (IDW) and kriging [11, 16] rely on assumptions of stationarity and Gaussian spatial statistics, which are often violated by the highly non-linear, localized nature of convective rainfall. More recent learning-based approaches, including convolutional networks [14, 21] and heterogeneous graph neural networks [18, 20], relax these assumptions by learning non-linear relationships directly from data and by integrating multiple sensing modalities. However, these methods typically reconcile heterogeneous sensors as feature channels or node types, without explicitly modeling the spatial support of each observation—whether it constrains the rainfall field at a point, along a line, or over an area.

We propose a geometry-aware multi-support Heterogeneous Graph Neural Network (HGNN) for rainfall-field reconstruction. The system represents gauges, CMLs, and

1

gridded remote-sensing products according to their measurement support, and fuses them through cross-support message passing into a point-support prediction layer. This allows rainfall to be queried at unobserved locations while preserving the geometry by which each sensor constrains the field.

On the real-world Singapore testbed, the proposed support-aware model reduces root mean square error (RMSE) by 23.2% over IDW and outperforms convolutional fusion and support-agnostic graph baselines.

Our contributions are:

* A practical multi-support rainfall reconstruction system that fuses point rain gauges, line-supported CMLs, and gridded radar/satellite products in a single geometry-aware graph.

* A support-aware graph construction that represents 0D point, 1D line, and 2D gridded observations as distinct support-typed layers and routes information through cross-support message passing.

* An inductive masked-node prediction protocol that prevents gauge-value leakage and enables rainfall estimates at unobserved target locations and user-defined output grids.

* A two-testbed application evaluation across Singapore and Sydney, showing that explicit support modeling improves reconstruction in under-sampled rainfall regimes and characterizing when multi-support fusion provides limited additional gain.

# 2. Related Work

## 2.1. Multi-Source Spatial Field Estimation

Estimating a continuous rainfall field from sparse, heterogeneous sensors is a long-standing problem. Classical approaches include deterministic interpolation, geostatistical interpolation, radar bias adjustment, and radar-gauge integration through methods such as mean-field bias adjustment, kriging with external drift, and Bayesian merging among the most widely used techniques [16]. Variational schemes that jointly retrieve rainfall from gauges, radar and microwave links have likewise shown that combining sources of differing character improves the resulting estimate [3]. Many of these methods rely on stationarity, smoothness, or Gaussian spatial-statistical assumptions that can be violated by localized convective rainfall, and they tend to degrade in precisely the sparse, localized regimes where fusion is most valuable [16].

Learned methods relax these assumptions by fitting non-linear mappings directly from data. Random-forest regression and convolutional networks have been applied to multi-source precipitation fusion [2, 7, 14], with random-forest merging combining gauges, gridded products, and topography in data-scarce regions [2], and convolutional fusion

of radar, satellite, and gauges yielding the largest gains where gauge coverage is sparse [14]. Within graph learning, Graph Neural Networks (GNNs) have been applied to radar-based quantitative precipitation estimation [18], but predominantly on homogeneous graphs in which all sensors are treated as equivalent nodes. Across this progression of geostatistical, convolutional, and graph-based methods, sensor heterogeneity is handled in *feature space*: each source contributes a different channel or node-feature vector, while the geometry of the measurement itself is left unmodelled. A gauge, a microwave link and a radar/satellite pixel are not interchangeable point samples; they constrain the underlying field over a point, a line and an area respectively, and the existing fusion methods do not represent this distinction in measurement support.

## 2.2. Heterogeneous and Off-Grid Graph Learning

HGNNs generalise message passing to graphs with multiple node and edge types $G = (V, E, \tau, \phi)$, where the type maps $\tau$ and $\phi$ enable type-specific transformations and metapath-based relational semantics [20]. GNNs have proven effective for large-scale geospatial modeling, including global weather emulation [12]. Standard HGNNs, however, distinguish sensors by node or edge *type* (which variables a node carries) without modeling the spatial support over which each sensor constrains the field. Heterogeneity of feature space is represented; heterogeneity of measurement geometry is not. Inductive graph formulations have been developed for spatiotemporal kriging, recovering signals at unsampled locations by masking nodes and reconstructing them on sampled subgraphs [22]; these operate on homogeneous point sensors, with no notion of differing measurement support.

The closest prior work to ours is Yang et al. [23], who fuse off-grid weather stations with gridded numerical-weather products on a heterogeneous graph for localized weather forecasting. Station nodes and grid-cell nodes form two node types, and each station aggregates information from neighboring stations and grid cells through message passing. This is structurally close to the gauge-plus-grid component of our framework, and the nearest point of comparison in the literature. We differ on three axes. First, both of their node types are treated as point-supported in the graph: weather stations and grid cells are represented by their coordinates and connected by Euclidean nearest-neighbor edges. Their graph thus encodes heterogeneity of feature space but not of spatial support. Our framework introduces line-supported observations as a first-class support type, with edges constructed from the geometry of the link rather than from point-to-point distance. Second, Yang et al. [23] predict at the fixed locations of known training stations under a temporal split, and do not reconstruct the field at unobserved locations; whereas our inductive set-

2

ting evaluates spatially held-out locations and allows unseen target nodes to be inserted at inference on user-defined target grids. Third, their task is correction of an existing gridded forecast toward local observations, whereas ours is direct estimation of a spatial field from concurrent multi-support observations. We position our contribution as the multi-support abstraction these distinctions imply: a single graph in which point, line and gridded supports coexist and exchange messages, with measurement geometry preserved in the graph structure itself.

## 2.3. Line-Supported Observations

CMLs provide the canonical line-supported measurement. Each link measures the path-averaged attenuation of the signal between two telecommunication towers, yielding a near-surface, path-integrated rainfall estimate along its span [17]. In dense urban networks (Singapore among them), large numbers of links are already deployed by mobile operators, providing rainfall-relevant observations where gauges are sparse, at little additional cost [8]. CML observations are often converted through multi-step pipelines before fusion; each step can introduce uncertainty that propagates into downstream estimates [17]; coverage is also uneven, with gaps over water bodies, terrain, and restricted airspace [8]. These properties make CMLs both valuable and awkward for existing fusion methods: their line geometry does not fit the point-node assumption underlying prior graph-based approaches, and prior work typically reduces links to representative points or pre-converted rainfall values before fusion. In our framework, a CML is treated as a single line-supported observation parameterized by its two endpoints, with cross-support edges constructed from line geometry rather than from a collapsed midpoint representation.

## 3. Methodology

We consider the problem of inferring a continuous two-dimensional physical field from a heterogeneous collection of observations that measure the same underlying process through different *spatial supports*. The spatial support of an observation is the geometric region over which it constrains the latent field. We distinguish three support types. A point (0D) observation constrains the field at a single location. A line (1D) observation constrains a path integral of the field along a one-dimensional segment. A gridded (2D) observation constrains an areal average of the field over a regular cell. Observations of different support are complementary, but cannot be optimally fused by treating them as generic graph nodes that differ only in their feature vectors. This discards how each observation geometrically constrains the field.

We propose a geometry-aware multi-support HGNN that preserves these differences. The framework represents observations as nodes organized into support-typed layers, en-

codes each according to its measurement geometry, and fuses them via cross-support message passing into a designated point-support prediction layer, from which the field is decoded. Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote the heterogeneous graph. The node set $\mathcal{V} = \mathcal{V}_0 \cup \mathcal{V}_1 \cup \mathcal{V}_2$ is partitioned by spatial support into point, line, and gridded nodes. In the rainfall application, $\mathcal{V}_0$ contains rain gauges and target query locations, $\mathcal{V}_1$ contains CML line supports, and $\mathcal{V}_2$ contains radar or satellite grid cells. The edge set $\mathcal{E}$ is partitioned into within-support edges, which connect observation nodes that share a support type, and cross-support edges, which connect observation nodes to the prediction layer. Each support layer may itself contain multiple heterogeneous sources of the same dimensionality: two distinct gridded sources, for instance, each become a separate node type within the 2D support layer, each with their own learned message-passing parameters, but constructed using the same geometric procedure for cross-support edges. The framework is therefore modular: adding or removing a source requires only instantiating or dropping the corresponding node type and its cross-support edges, leaving the rest of the architecture unchanged. We exploit this in Section 4, where the same framework is instantiated on two testbeds with different support compositions.

## 3.1. Support-Typed Node Layers

Nodes in $\mathcal{G}$ are organised into three support layers, each encoding observations according to their measurement geometry.

**Point (0D) nodes.** Each point-support observation is represented as a single node $v \in \mathcal{V}_0$ at its geographical coordinate $x_v \in \mathbb{R}^2$. The 0D layer serves a dual role: observed nodes carry measurements of the latent field at known locations, while target nodes have their observations masked to zero and are the locations at which the field is to be predicted. Both are structurally identical 0D nodes: the masking is applied to node features at training time, not to the graph topology, so the target values must be inferred entirely from messages propagated by neighboring observed and cross-support nodes.

**Line (1D) nodes.** Each line-support sensor is represented as a pair of nodes $\{v_A, v_B\} \subset \mathcal{V}_1$ corresponding to its two physical endpoints. In the graph implementation, the line support is encoded by endpoint nodes tied by a shared measurement and a within-support edge. This preserves the line-support geometry explicitly; the sensor is not collapsed to a single proxy point such as its midpoint, but retained as two endpoint nodes spanning the full measurement path.

**Gridded (2D) nodes.** Each cell of a gridded field product is represented as a single node $v \in \mathcal{V}_2$ positioned at the cell centre. Where multiple gridded sources are present, each source occupies a distinct node type with its own learned message-passing parameters, but is constructed us-

3

ing the same geometric procedure for cross-support edges described in Section 3.2.

**Node features.** Point-support (0D) nodes carry the observed value together with a validity flag that distinguishes a genuine zero observation from a masked or missing one, and a Laplacian positional encoding (LPE) given by the four smallest non-trivial eigenvectors of the graph Laplacian [9]. The validity flag is necessary because a masked target node and a true dry reading both present a value of zero; encoding observation validity as an explicit feature follows prior work on learning from sparse and partially-observed spatiotemporal graphs [13]. The LPE assigns each node a structurally unique position in the graph, letting the model distinguish nodes that carry similar observed values but occupy different positions in the network. For line-supported CMLs, both endpoint nodes share the same link measurement, with the endpoints defining the path geometry rather than two independent point observations. Gridded nodes carry the corresponding radar or satellite cell value.

## 3.2. Edge Construction

Edges connect nodes either within a single support layer or across support layers into the prediction layer (Figure 1). Spatial proximity edges are weighted by normalized inverse distance. The internal edge connecting the two endpoints of a CML uses the physical path length as its line-support attribute. For a pair of nodes $i, j$ with Euclidean distance $d_{ij}$, the weight is normalised against the strongest connection at the target node,

$$ w_{ij} = \frac{1/d_{ij}}{\max_{k \in \mathcal{N}(j)} 1/d_{kj}}, \tag{1} $$

so that weights lie in $(0, 1]$ and the nearest neighbour of each target node receives unit weight. This normalisation is applied to all edge types described below.

**Within-support edges.** Point (0D) nodes are connected to their $k$ nearest point neighbors using a ball-tree search over node coordinates, with weights given by Eq. (1). Each line (1D) sensor contributes a single within-support edge joining its two endpoint nodes, weighted by physical length of the integration path. Gridded (2D) nodes are connected to their immediate neighbors on the regular grid, each cell linking to its surrounding cells in its local (eight-neighbor) neighborhood, so that within-support message passing propagates information across the gridded field.

**Cross-support edges.** Every observation layer is connected to the point-support prediction layer through cross-support edges, and these are the only paths by which line and gridded observations reach the prediction. For a gridded source, each prediction node is connected to its $k$ nearest gridded nodes under the haversine metric, weighted by Eq. (1).

engineering_drawing: Heterogeneous graph construction diagram showing 2D, 1D, and 0D support layers connecting to a 2D prediction layer.

Figure 1. (a) Heterogeneous graph construction over three measurement support types: 2D gridded nodes (red squares), 1D line nodes (green circles), and 0D point nodes (dark blue circles). Cross-support edges route information from 2D and 1D nodes into the 0D prediction layer. (b) At inference, a virtual node $v$ is inserted at any target location and connected to the existing graph via cross-support edges (zoom inset). Repeating this process across a regular grid yields a dense predicted field at a user-defined display resolution (fine, left; coarse, right).

For a line source, connectivity respects the line geometry of the sensor rather than approximating it by a point. For a prediction node $v$ and a line sensor with segment geometry $\ell$ (where $p$ denotes any point along the link path), the selection distance is the perpendicular distance from the node to the segment,

$$ d_{\perp}(v, \ell) = \min_{p \in \ell} \|x_v - p\|_2, \tag{2} $$

and the $k$ line sensors minimising Eq. (2) are selected for each prediction node. The node is then connected to both endpoint nodes of each selected sensor, with each endpoint edge weighted by Eq. (1) using its own node-to-endpoint distance. A prediction node therefore receives a separate, distance-appropriate message from each end of the integration path, preserving the line support of the observation rather than collapsing it to the midpoint.

**Leakage Control.** Cross-support edges are directed into the prediction layer, and all edges are reconstructed independently for each data split. Target nodes held out for evaluation never enter the training-graph topology, so no information about their location or connectivity is available at training time.

## 3.3. Message Passing

Given the constructed graph, the prediction at each point-support node is computed by message passing over the heterogeneous edge set. We use a heterogeneous convolution in which a separate graph convolution is instantiated for every edge type, so that messages arriving along different support relations are transformed by their own learned parameters before being combined, implemented with the heterogeneous convolution wrapper of PyTorch Geometric [10]. For a target node $v$ at layer $l$, each incoming edge type $r$ produces an aggregated message, and the messages from all

4

edge types incident on $v$ are combined by mean aggregation:

$$ h_v^{(l+1)} = \sigma \left( \underset{r \in \mathcal{R}(v)}{\text{mean}} \sum_{j \in \mathcal{N}_r(v)} w_{jv} h_j^{(l)} \mathbf{W}_r^{(l)} \right), \quad (3) $$

where $\mathcal{R}(v)$ is the set of edge types incident on $v$, $\mathcal{N}_r(v)$ is the neighbourhood of $v$ under edge type $r$, $w_{jv}$ is the inverse-distance edge weight from Eq. (1), $\mathbf{W}_r^{(l)}$ is the learned weight matrix for edge type $r$ at layer $l$, and $\sigma$ is a non-linear activation. A shared hidden dimension is used across all node types so that embeddings from different support layers are directly comparable when aggregated at a prediction node.

Stacking $L$ such layers allows each prediction node to aggregate information from progressively larger neighbourhoods: cross-support edges admit line and gridded observations into the point-support layer, and within-support edges then propagate this fused information between point nodes. Each hidden layer applies a ReLU non-linearity, while the final linear layer maps the embedding of each point-support node to a scalar field estimate with no output activation. Since rainfall is non-negative, predictions are clamped to be non-negative at evaluation time rather than constrained during training; we found that avoiding an output activation during training prevents vanishing gradients across many dry periods in the record. Because a distinct weight matrix $\mathbf{W}_r^{(l)}$ is learned per edge type, the model can weight each support relation differently rather than treating all incoming observations identically, while the shared aggregation keeps the architecture invariant to the number and composition of support layers present.

## 3.4. Inductive Training and Field Reconstruction

**Leave-one-out masked training.** A difficulty specific to this setting is that point-support observations serve as both input features and prediction targets. Without intervention, the model can learn a trivial identity mapping that copies each node's own observation to its output, minimizing training loss without learning any spatial relationship from its neighbors. To prevent this, we adopt a leave-one-out masking scheme, following the inductive mask-and-reconstruct principle used in graph-based kriging and imputation [13, 22]. At each training step, a single point-support node has its observation masked, and the model predicts that node's value from its neighbors only. Crucially, only the observed value and validity flag are masked; the Laplacian positional encoding is preserved, so the masked node retains its structural identity in the graph while losing its observed value. The loss is computed solely on the masked node,

$$ \mathcal{L} = (\hat{y}_v - y_v)^2, \quad (4) $$

where $y_v$ is the observed rainfall at the masked point-support node $v$, and $\hat{y}_v$ is the corresponding model prediction. This masking-and-prediction step is repeated once for every point-support node in the graph, with gradients accumulated across all such passes before a single parameter update. No node is ever exposed to its own ground-truth value while learning to predict it. At evaluation time, all held-out nodes are masked simultaneously in a single pass, since their predictions are mutually independent given the observed nodes and carry no risk of leakage between one another.

**Leakage control through graph reconstruction.** The graph is rebuilt independently for each data split. Held-out nodes enter only the evaluation graph, never the training graph, so no information about their location or connectivity influences training. Normalisation statistics are likewise computed on the training split alone.

**Field reconstruction on user-defined target grids.** Because the model predicts at a masked point-support node using only messages from its neighbors, the prediction is not tied to the locations observed during training. At inference, a regular grid of target nodes is inserted into the point-support layer. Each target node is connected to nearby observed point nodes through within-support edges and to nearby line and gridded observations through the same cross-support edge rules used during training. A single forward pass then yields a field estimate at every target node. The graph is therefore dynamic at inference: the resolution of the reconstructed field is determined by the density of inserted target nodes and is decoupled from the resolution of any input source. The same trained model can be queried on user-defined target grids without retraining.

## 4. Experimental Setup

We evaluate the framework on rainfall estimation, a problem that is a natural instantiation of multi-support fusion. Rainfall is one of the few physical fields observed simultaneously through all three support types: rain gauges provide accurate point (0D) measurements at fixed locations; CMLs provide path-integrated (1D) observations of attenuation along communication paths; and weather radar and satellite products provide gridded (2D) estimates over regular cells. These sources are individually limited and mutually complementary: gauges are precise but spatially sparse, gridded products offer broad coverage but are biased and indirect, and links offer near-surface path-averaged information between the two, which makes their fusion a demanding test of whether geometry-aware multi-support modeling improves the reconstruction of a single underlying field.

## 4.1. Datasets

We instantiate the framework on two geographically and climatically distinct testbeds with different support com-

5

positions, demonstrating that the same architecture applies without modification across sensor configurations.

**Singapore (0D + 1D + 2D).** The Singapore testbed combines all three support types over the period 17 May 2024 to 30 April 2025. It comprises 70 rain gauge stations (0D) regulated by the National Environment Agency [15], 414 CMLs (1D), and weather radar (2D) at 1 km resolution, the latter two provided by Pluvia.ai [19]. The radar product spans a $240 \times 240$ km domain; evaluation is performed at the gauge stations over the Singapore landmass (approximately $50 \times 27$ km).

**Sydney (0D + 2D + 2D).** The Sydney testbed combines point and two distinct gridded supports over the Greater Sydney region (approximately $70 \times 55$ km) for 2022 at hourly resolution. It comprises 75 gauge stations (0D) [4], gauge-corrected weather radar (2D) from the operational Rainfields 3 product [5] subsampled to approximately 5 km spacing, and satellite retrievals (2D) from the Himawari-8/9 CRRPH product [6] at 2 km resolution. This composition exercises the framework’s ability to fuse multiple heterogeneous gridded sources within the 2D support layer, each represented as a distinct node type sharing the same cross-support construction.

## 4.2. Evaluation Protocol

We evaluate every method under five-fold spatial cross-validation, in which held-out gauge stations are masked and predicted from the remaining observations. Stations are partitioned into spatially coherent folds so that held-out stations are not adjacent to training stations, preventing a model from succeeding through trivial spatial proximity rather than genuine generalisation. The same held-out stations are used across all methods and all support configurations to ensure comparability. Normalisation statistics are computed on the training split alone and applied to the held-out stations. Predictions are evaluated against the held-out gauge observations using RMSE and Pearson correlation ($r$), reported as the mean across folds.

## 4.3. Implementation Details

Singapore experiments use 15-minute data, with gauge readings resampled from 5-minute accumulations to align with CML and radar timestamps; Sydney experiments use hourly data. Graphs are built with $k$-nearest-neighbor search over sensor coordinates, using $k = 5$ for point-point and CML-point connections and $k = 9$ for grid-point connections. Features are z-score normalized per node type using training-split statistics only. The HGNN uses five message-passing layers, hidden dimension 32, and dropout $p = 0.15$. All neural models are trained with Adam

($3 \times 10^{-4}$ learning rate, $10^{-4}$ weight decay), weighted MSE loss, and early stopping with patience 10.

## 4.4. Baselines

We compare against three tiers of baseline, each isolating a different aspect of the contribution.

**Classical interpolation.** IDW represents the geometry-agnostic, non-learned interpolation methods in operational use, and establishes the performance floor that any learned method must exceed. Among classical statistical interpolators, geostatistical kriging variants such as ordinary kriging and kriging with external drift incorporate spatial covariance structure and auxiliary predictors such as elevation [11, 16]. In preliminary evaluation on our data both variants underperformed IDW, consistent with the known degradation of kriging in sparse, non-stationary regimes [16]; we therefore adopt IDW as the best-performing classical baseline.

**Learned non-graph fusion.** Convolutional neural networks represent learned fusion without graph structure, operating on gridded inputs rather than an explicit graph. We adapt the spatiotemporal fusion architecture of Wu et al. [21], which extracts spatial features from co-located gridded sources through per-source convolutional branches and predicts rainfall at gauge locations. The inputs are three gridded channels (an inverse-distance interpolated gauge field, gridded CML, and gridded radar), and the model is trained and evaluated under the same spatial cross-validation protocol as the graph models. We report two variants from this family: a single-frame convolutional model (CNN) that fuses the gridded sources at a single time step, and a convolutional-recurrent model (CNN-LSTM) that adds an LSTM to capture temporal dependence across consecutive frames. Together these isolate the value of graph structure: both see the same observations as our model but represent them on a fixed grid rather than as geometry-aware support-typed nodes.

**Support-agnostic HGNN.** The most direct ablation of the contribution. This baseline uses an identical architecture, node features, capacity, training scheme, and data, but represents all observations as generic point nodes: gridded cells and link endpoints are inserted as ordinary 0D nodes connected by plain distance-based edges, with a single shared convolution in place of support-typed message passing. It retains the same observational information as our model but discards the support geometry. Improvements over this baseline isolate the effect of support-aware construction as much as possible while holding model capacity and inputs fixed.

6

Table 1. Reconstruction performance at held-out gauge locations on the Singapore testbed (0D, 1D, and 2D supports). RMSE in mm h<sup>−1</sup> and Pearson $r$ against withheld gauge values; $\Delta\%$ is relative change versus the IDW baseline. The convolutional baseline consumes the sources as gridded channels. Supports: G (gauge), C (CML), R (radar). Evaluation follows the spatial cross-validation protocol of Section 4.2. Best in bold.

<table>
  <thead>
    <tr>
        <th>Model</th>
        <th>RMSE ↓</th>
        <th>Δ% vs IDW</th>
        <th>Pearson r ↑</th>
        <th>Δ% vs IDW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>IDW (baseline)</td>
        <td>2.429</td>
        <td>—</td>
        <td>0.714</td>
        <td>—</td>
    </tr>
    <tr>
        <td>CNN (G+C+R)</td>
        <td>2.254</td>
        <td>−7.2</td>
        <td>0.697</td>
        <td>−2.4</td>
    </tr>
    <tr>
        <td>CNN-LSTM (G+C+R)</td>
        <td>2.047</td>
        <td>−15.7</td>
        <td>0.744</td>
        <td>+4.2</td>
    </tr>
    <tr>
        <td>HGNN (sup. agnostic)</td>
        <td>2.130</td>
        <td>−12.3</td>
        <td>0.737</td>
        <td>+3.2</td>
    </tr>
    <tr>
        <td>HGNN (G)</td>
        <td>1.946</td>
        <td>−19.9</td>
        <td>0.780</td>
        <td>+9.2</td>
    </tr>
    <tr>
        <td>HGNN (G+C)</td>
        <td>1.927</td>
        <td>−20.7</td>
        <td>0.789</td>
        <td>+10.4</td>
    </tr>
    <tr>
        <td>HGNN (G+C+R)</td>
        <td><strong>1.866</strong></td>
        <td><strong>−23.2</strong></td>
        <td><strong>0.793</strong></td>
        <td><strong>+11.1</strong></td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
        <th colspan="3">Spatial Reconstruction Comparison at 2025-04-20 17:00</th>
    </tr>
    <tr>
        <th>(a) Radar reflectivity</th>
        <th>(b) Gauge-only IDW</th>
        <th>(c) Multi-support HGNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>[visual representation]</td>
        <td>[visual representation]</td>
        <td>[visual representation]</td>
    </tr>
  </tbody>
</table>

Figure 2. (a) Radar reflectivity (spatial reference only), (b) gauge-only IDW, and (c) multi-support HGNN (gauge + CML + radar) prediction at 2025-04-20 17:00, Singapore.

# 5. Results and Discussion

## 5.1. Multi-Support Fusion on the Singapore Testbed

Table 1 reports reconstruction performance at held-out gauge locations on the Singapore testbed, where 0D (gauge), 1D (CML), and 2D (radar) supports are available. Each model predicts at gauge locations held out during evaluation and is scored against the withheld observations using RMSE (mm h<sup>−1</sup>) and Pearson correlation, with relative change against the IDW baseline.

The following findings stand out. First, the learned graph model improves substantially over classical interpolation even using gauges alone: the gauge-only HGNN reduces RMSE by 19.9% (1.946 versus 2.429) and raises correlation by 9.2% (0.780 versus 0.714) relative to IDW.

Second, performance improves monotonically as supports of differing geometry are added: introducing the 1D CML support lowers RMSE to 1.927, and adding the 2D radar support lowers it further to 1.866, a 23.2% reduction over IDW and the best result in the table. This progression is the central evidence for the multi-support hypothesis: each additional support type, with its own geometry encoded in the graph, contributes information that the others do not.

Third, the most telling finding for the framing of this work, the single-frame convolutional baseline (CNN) that consumes the same gridded inputs as stacked image channels attains an RMSE of only 2.254, a mere 7.2% improvement over IDW and worse than even the gauge-only HGNN. Adding temporal modeling (CNN-LSTM) helps, lowering RMSE to 2.047 and lifting correlation above IDW, but it still trails every HGNN variant, including the gauge-only model that uses strictly less information. The gridded sources carry the same raw information in both the convolutional and graph models; the difference is that the CNN treats them as dense image channels registered to a pixel grid, discarding their geometric relationship as 2D supports to the underlying field and to the off-grid 0D and 1D observations. The support-aware graph, which preserves that geometry, extracts markedly more from the identical inputs.

Figure 2 illustrates this mechanism at a single convective timestep. Radar is shown as a spatial structural reference only; its magnitude is biased relative to gauge observations and the panels use independent scales. Radar resolves a compact, off-gauge cell that gauge-only IDW cannot recover, smoothing the field into broad structure around the gauge locations; the multi-support HGNN, fusing the same radar with gauges and links, reconstructs the cell while remaining calibrated to gauge magnitudes. Radar magnitude is biased but is accepted as a reliable indicator of spatial structure [[16]](16), and fusing it recovers structure that gauge-only interpolation cannot. The reconstructed fields shown are produced at a chosen display resolution by the same trained model, by inserting target nodes on a regular grid (Section 3); the resolution is a free parameter at inference rather than a property of any input source.

**Support-agnostic baseline.** Fourth, the gain is largely associated with support-aware construction rather than graph structure alone. A support-agnostic HGNN, identical in architecture, features, capacity, and training but representing every observation as a 0D node connected by plain distance-based edges, attains an RMSE of 2.130, recovering only part of the gap between IDW and the full multi-support model. Stripping the support typing while holding everything else fixed forfeits roughly half of the improvement the support-typed model achieves over IDW (a 12.3% reduction versus 23.2%). Where the convolutional baseline shows that images discard geometry, this baseline shows that a graph alone is not sufficient: the same observations, given a graph but stripped of their support typing, leave much of the available skill unrecovered. This comparison indicates that much of the gain comes from support-aware construction rather than graph structure alone.

## 5.2. Generalization to the Sydney Testbed

The Singapore results show large and consistent gains. To probe the boundary of where these gains hold, we apply the

7

Table 2. Generalization to the Sydney testbed (point and two gridded supports). RMSE in mm h<sup>−1</sup> and Pearson $r$ against withheld gauge values; $\Delta\%$ is relative change versus the IDW baseline. Evaluation follows the protocol of Section 4.2. Supports: G (gauge), R (radar), S (satellite). Best in bold.

<table>
  <thead>
    <tr>
        <th>Model</th>
        <th>RMSE ↓</th>
        <th>$\Delta\%$ vs IDW</th>
        <th>Pearson $r \uparrow$</th>
        <th>$\Delta\%$ vs IDW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>IDW (baseline)</td>
        <td><strong>0.667</strong></td>
        <td>—</td>
        <td><strong>0.854</strong></td>
        <td>—</td>
    </tr>
    <tr>
        <td>HGNN (G)</td>
        <td>0.686</td>
        <td>+2.9</td>
        <td>0.843</td>
        <td>−1.3</td>
    </tr>
    <tr>
        <td>HGNN (G+R)</td>
        <td>0.688</td>
        <td>+3.2</td>
        <td>0.842</td>
        <td>−1.4</td>
    </tr>
    <tr>
        <td>HGNN (G+R+S)</td>
        <td>0.698</td>
        <td>+4.8</td>
        <td>0.840</td>
        <td>−1.6</td>
    </tr>
  </tbody>
</table>

same framework to a Sydney testbed with different support configuration (0D and two 2D sources, radar and satellite; no 1D support) and a different rainfall regime, evaluated at held-out gauge locations under the same spatial cross-validation protocol. Table 2 reports the outcome: the gauge-only HGNN does not improve over IDW (correlation 0.843 versus 0.854; RMSE 0.686 versus 0.667), and adding the 2D radar and satellite supports does not change this picture (correlation 0.842 with radar, 0.840 with radar and satellite).

Rather than treat this as an isolated negative result, we find that a single mechanism explains both the Singapore success and the Sydney plateau: the skill available to a learned or fused estimator appears to depend on the gauge spacing *relative to the spatial correlation length of the rainfall field*, not by gauge count. When a field is sampled finely relative to its correlation length, it is already well determined by the gauges; little residual structure remains for either a learned model or an additional gridded sensor to recover. When a fast-decorrelating field is sampled coarsely relative to its correlation length, substantial sub-gauge structure remains, and both learning and fusion have headroom.

The two testbeds sit on opposite sides of this divide, and the baseline correlations reveal it directly. Despite a *denser* gauge network, Singapore’s IDW baseline is markedly weaker than Sydney’s (correlation 0.714 versus 0.854): interpolation skill appears to depend on the field’s spatial correlation, not the gauge count. Sydney’s rainfall remains coherent across its gauge spacing: empirical inter-gauge Pearson correlation (computed from pairwise gauge correlations on the full gauge record) is approximately 0.85 at the ~3 km median spacing with a length near 14 km, so a held-out location is already well predicted by its neighbours, leaving little for the model to add and IDW close to optimal. Singapore’s convective rainfall, by contrast, decorrelates over a much shorter distance, leaving sub-gauge structure that fixed interpolation cannot resolve and that the support-aware model recovers.

This also explains why adding Sydney’s gridded products does not improve the estimate as Table 2 shows. When nearby gauges already provide a low-uncertainty estimate, coarser radar or satellite observations may add bias rather than useful residual structure. In Singapore, by contrast, the under-sampled convective field leaves sub-gauge structure that radar can help localize.

We note two honest limitations of this argument. The Singapore and Sydney testbeds differ along two axes simultaneously (Singapore additionally possesses a 1D (CML) support and exhibits faster decorrelation) and our data cannot fully separate the two; we therefore describe the correlation-length account as consistent with the evidence rather than proven. The gauge-only comparison, in which no 1D support is involved, provides the cleaner support for the mechanism, since there the entire gap must be attributed to the field’s spatial correlation rather than to support complementarity. Taken together, the two testbeds delineate the operating envelope of multi-support fusion: it delivers the largest gains where supports are geometrically complementary and the field is under-sampled relative to its correlation length, and offers little where the field is already well resolved by existing 0D observations.

# 6. Conclusion

We introduced a geometry-aware multi-support HGNN that represents observations according to their spatial support, point, line or gridded, and fuses them through cross-support message passing into a shared prediction layer. By making the structural distinction between 0D, 1D, and 2D supports explicit in the graph rather than absorbing it into feature space, the framework treats each measurement as the geometric constraint it actually imposes on the field. An inductive masked-node formulation lets a single trained model reconstruct the field at user-defined target grids, decoupled from the resolution of any input source.

On rainfall estimation, encoding support geometry improved reconstruction RMSE by 23.2% over classical interpolation and outperformed both the convolutional fusion baseline and a support-agnostic graph of matched capacity, indicating that support-aware construction, rather than graph structure alone, accounts for much of the gain. A second testbed with a different support composition showed where the approach helps and where it does not: multi-support fusion delivers the largest gains when the field is under-sampled relative to its spatial correlation length, and little when existing point observations already resolve the field. This account of when fusion helps, rather than a single benchmark number, is what we expect to carry over to other multi-support estimation problems.

The framework is limited to the support types instantiated here, and the two testbeds differ along more than one axis, so the correlation-length account is consistent with our evidence rather than proven. Future work will evaluate ad-

8

ditional cities, richer line- and area-support encodings, temporal forecasting, and downstream hydrological impact.

# References

[1] Carlo Malapad Acosta, Herath Mudiyanselage Viraj Vidura Herath, Jia Yu Lim, Abhishek Saha, Sanka Rasnayaka, and Lucy Amanda Marshall. Physics-informed graph neural networks for operational flood modeling. *ArXiv*, abs/2512.23964, 2025. 1

[2] Oscar Baez-Villanueva, Mauricio Zambrano-Bigiarini, Hylke Beck, Ian McNamara, Lars Ribbe, Alexandra Nauditt, Christian Birkel, Koen Verbist, Juan Diego Giraldo-Osorio, and Nguyen Thinh. Rf-mep: A novel random forest method for merging gridded precipitation products and ground-based measurements. *Remote Sensing of Environment*, 239, 2020. 2

[3] Blandine Bianchi, Peter Jan Van Leeuwen, Robin Hogan, and Alexis Berne. A variational approach to retrieve rain rate by combining information from rain gauges, radars, and microwave links. *Journal of Hydrometeorology*, 14:1897–1909, 2013. 2

[4] Bureau of Meteorology. WaterData: Water data online. [http://www.bom.gov.au/waterdata/](http://www.bom.gov.au/waterdata/). Retrieved February 2026. 6

[5] Bureau of Meteorology. AURA – operational radar Rainfields 3, 2022. Accessed via NCI project rq0. 6

[6] Bureau of Meteorology. Himawari 8/9 convective rain rate – physically based (CRRPH), 2024. Accessed via NCI project rv74. 6

[7] Hanqing Chen, Jiangyuan Zeng, Yi Lyu, and Bin Yong. Multisource precipitation data fusion: Generating high-quality precipitation estimates. *The Innovation Geoscience*, 4: 100195, 2026. 2

[8] Christian Chwala and Harald Kunstmann. Commercial microwave link networks for rainfall observation: Assessment of the current status and future challenges. *Wiley Interdisciplinary Reviews: Water*, 6, 2019. 3

[9] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. *CoRR*, abs/2012.09699, 2020. 4

[10] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. *CoRR*, abs/1903.02428, 2019. 4

[11] Pierre Goovaerts. Geostatistical approaches for incorporating elevation into the spatial interpolation of rainfall. *Journal of Hydrology*, 228:113–129, 2000. 1, 6

[12] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, Alexander Merose, Stephan Hoyer, George Holland, Oriol Vinyals, Jacklynn Stott, Alexander Pritzel, Shakir Mohamed, and Peter Battaglia. Learning skillful medium-range global weather forecasting. *Science (New York, N.Y.)*, 382: eadi2336, 2023. 2

[13] Ivan Marisca, Andrea Cini, and Cesare Alippi. Learning to reconstruct missing data from spatiotemporal graphs with sparse observations, 2022. 4, 5

[14] Arthur Moraux, Steven Dewitte, Bruno Cornelis, and Adrian Munteanu. A deep learning multimodal method for precipitation estimation. *Remote Sensing*, 13:3278, 2021. 1, 2

[15] National Environment Agency. Rainfall across Singapore. data.gov.sg, Dataset, 2024. Retrieved April 8, 2026. 6

[16] Susana Ochoa-Rodriguez, Li-Pen Wang, Patrick Willems, and Christian Onof. A review of radar-rain gauge data merging methods and their potential for urban hydrological applications. *Water Resources Research*, 55, 2019. 1, 2, 6, 7

[17] A. Overeem, H. Leijnse, and Remko Uijlenhoet. Retrieval algorithm for rainfall mapping from microwave links in a cellular communication network. *Atmospheric Measurement Techniques Discussions*, 8:8191–8230, 2015. 3

[18] Xuan Peng, Qian Li, and Jinrui Jing. Cngat: A graph neural network model for radar quantitative precipitation estimation. *IEEE Transactions on Geoscience and Remote Sensing*, PP:1–1, 2021. 1, 2

[19] Pluvia.ai. Pluvia – commercial microwave link and radar data. CML and gauge-corrected radar data for Singapore, provided via research collaboration, 2025. 6

[20] Xiao Wang, Deyu Bo, Chuan Shi, Shaohua Fan, Yanfang Ye, and Philip S. Yu. A survey on heterogeneous graph embedding: Methods, techniques, applications and sources. *IEEE Transactions on Big Data*, 9:415–436, 2020. 1, 2

[21] Hongcai Wu, Qinli Yang, Jiaming Liu, and Guoqing Wang. A spatiotemporal deep fusion model for merging satellite and gauge precipitation in china. *Journal of Hydrology*, 584: 124664, 2020. 1, 6

[22] Yuankai Wu, Dingyi Zhuang, Aurelie Labbe, and Lijun Sun. Inductive graph neural networks for spatiotemporal kriging, 2020. 2, 5

[23] Qidong Yang, Jonathan Giezendanner, Daniel Salles Civitarese, Johannes Jakubik, Eric Schmitt, Anirban Chandra, Jeremy Vila, Detlef Hohl, Chris Hill, Campbell Watson, and Sherrie Wang. Local off-grid weather forecasting with multi-modal earth observation data. *Journal of Advances in Modeling Earth Systems*, 17(12):e2025MS005207, 2025. e2025MS005207 2025MS005207. 2

9
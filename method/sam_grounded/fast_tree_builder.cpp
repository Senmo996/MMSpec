#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Path = std::vector<int>;
using Edge = std::pair<std::int64_t, std::int64_t>;

struct CandidateDistribution {
  std::vector<std::int64_t> candidates;
  std::vector<double> probabilities;
};

struct SelectedNode {
  std::int64_t token = 0;
  int depth = 0;
  int rank = 0;
  Path rank_path;
  Path parent_path;
  std::int64_t previous_token = -1;
  std::int64_t previous_previous_token = -1;
  std::int64_t previous_previous_previous_token = -1;
  std::set<Edge> path_edges;
};

struct PendingNode {
  double negative_score = 0.0;
  Path rank_path;
  std::int64_t candidate = 0;
  Path parent_path;
  std::set<Edge> path_edges;
};

struct PendingGreater {
  bool operator()(const PendingNode& lhs, const PendingNode& rhs) const {
    if (lhs.negative_score != rhs.negative_score) {
      return lhs.negative_score > rhs.negative_score;
    }
    if (lhs.rank_path != rhs.rank_path) {
      return lhs.rank_path > rhs.rank_path;
    }
    if (lhs.candidate != rhs.candidate) {
      return lhs.candidate > rhs.candidate;
    }
    return lhs.parent_path > rhs.parent_path;
  }
};

CandidateDistribution read_distribution(
    const py::dict& candidate_rows,
    const py::dict& score_rows,
    const py::object& key,
    int width) {
  CandidateDistribution result;
  if (!candidate_rows.contains(key)) {
    return result;
  }

  py::sequence candidates = candidate_rows[key].cast<py::sequence>();
  const int candidate_count = std::min<int>(py::len(candidates), width);
  result.candidates.reserve(candidate_count);
  for (int index = 0; index < candidate_count; ++index) {
    result.candidates.push_back(
        py::cast<std::int64_t>(candidates[index]));
  }

  if (score_rows.contains(key)) {
    py::sequence scores = score_rows[key].cast<py::sequence>();
    if (py::len(scores) >= candidate_count) {
      result.probabilities.reserve(candidate_count);
      for (int index = 0; index < candidate_count; ++index) {
        result.probabilities.push_back(py::cast<double>(scores[index]));
      }
    }
  }
  if (result.probabilities.size() < result.candidates.size()) {
    result.probabilities.clear();
    result.probabilities.reserve(candidate_count);
    double total = 0.0;
    for (int rank = 1; rank <= candidate_count; ++rank) {
      const double value = 1.0 / static_cast<double>(rank);
      result.probabilities.push_back(value);
      total += value;
    }
    if (total == 0.0) {
      total = 1.0;
    }
    for (double& value : result.probabilities) {
      value /= total;
    }
  }
  return result;
}

CandidateDistribution candidate_distribution(
    std::int64_t parent_token,
    std::int64_t previous_token,
    std::int64_t previous_previous_token,
    int width,
    const py::dict& unigram_rows,
    const py::dict& unigram_scores,
    const py::dict& context_rows,
    const py::dict& context_scores,
    const py::dict& trigram_rows,
    const py::dict& trigram_scores) {
  std::vector<CandidateDistribution> sources;
  sources.reserve(3);

  if (previous_previous_token >= 0 && previous_token >= 0) {
    auto distribution = read_distribution(
        trigram_rows,
        trigram_scores,
        py::make_tuple(
            previous_previous_token, previous_token, parent_token),
        width);
    if (!distribution.candidates.empty()) {
      sources.push_back(std::move(distribution));
    }
  }
  if (previous_token >= 0) {
    auto distribution = read_distribution(
        context_rows,
        context_scores,
        py::make_tuple(previous_token, parent_token),
        width);
    if (!distribution.candidates.empty()) {
      sources.push_back(std::move(distribution));
    }
  }
  {
    auto distribution = read_distribution(
        unigram_rows,
        unigram_scores,
        py::int_(parent_token),
        width);
    if (!distribution.candidates.empty()) {
      sources.push_back(std::move(distribution));
    }
  }

  if (sources.empty()) {
    return {};
  }
  if (sources.size() == 1) {
    return std::move(sources.front());
  }

  static constexpr std::array<double, 4> source_weights = {
      0.70, 0.20, 0.10, 0.05};
  std::vector<std::int64_t> fused_candidates;
  std::vector<double> fused_scores;
  std::unordered_map<std::int64_t, std::size_t> candidate_indices;
  fused_candidates.reserve(sources.size() * width);
  fused_scores.reserve(sources.size() * width);
  candidate_indices.reserve(sources.size() * width);

  for (std::size_t source_index = 0; source_index < sources.size();
       ++source_index) {
    const double weight = source_weights[std::min<std::size_t>(
        source_index, source_weights.size() - 1)];
    const auto& source = sources[source_index];
    for (std::size_t index = 0; index < source.candidates.size(); ++index) {
      const auto candidate = source.candidates[index];
      auto [iterator, inserted] = candidate_indices.emplace(
          candidate, fused_candidates.size());
      if (inserted) {
        fused_candidates.push_back(candidate);
        fused_scores.push_back(0.0);
      }
      fused_scores[iterator->second] += weight * source.probabilities[index];
    }
  }

  std::vector<std::size_t> ordering(fused_candidates.size());
  for (std::size_t index = 0; index < ordering.size(); ++index) {
    ordering[index] = index;
  }
  std::stable_sort(
      ordering.begin(), ordering.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (fused_scores[lhs] != fused_scores[rhs]) {
          return fused_scores[lhs] > fused_scores[rhs];
        }
        return lhs < rhs;
      });
  if (ordering.size() > static_cast<std::size_t>(width)) {
    ordering.resize(width);
  }

  CandidateDistribution result;
  result.candidates.reserve(ordering.size());
  result.probabilities.reserve(ordering.size());
  double total = 0.0;
  for (const std::size_t index : ordering) {
    result.candidates.push_back(fused_candidates[index]);
    result.probabilities.push_back(fused_scores[index]);
    total += fused_scores[index];
  }
  if (total == 0.0) {
    total = 1.0;
  }
  for (double& value : result.probabilities) {
    value /= total;
  }
  return result;
}

}  // namespace

std::vector<std::array<std::int64_t, 4>> build_score_priority_nodes(
    std::int64_t root_token,
    std::int64_t root_previous_token,
    std::int64_t root_previous_previous_token,
    int width,
    int branch_width,
    int depth,
    int node_budget,
    std::int64_t blocked_token_id,
    const py::dict& unigram_rows,
    const py::dict& unigram_scores,
    const py::dict& context_rows,
    const py::dict& context_scores,
    const py::dict& trigram_rows,
    const py::dict& trigram_scores,
    const std::vector<double>& hit_masses) {
  if (width <= 0 || branch_width <= 0 || depth <= 0 || node_budget <= 0) {
    return {};
  }
  if (hit_masses.empty()) {
    throw std::invalid_argument("hit_masses must not be empty");
  }

  std::map<Path, SelectedNode> selected;
  SelectedNode root;
  root.token = root_token;
  root.previous_token = root_previous_token;
  root.previous_previous_token = root_previous_previous_token;
  selected.emplace(Path{}, root);

  std::vector<SelectedNode> selected_nodes;
  selected_nodes.reserve(node_budget);
  std::priority_queue<
      PendingNode, std::vector<PendingNode>, PendingGreater>
      pending;

  auto push_children = [&](const Path& parent_path, double parent_score) {
    const int current_depth = static_cast<int>(parent_path.size()) + 1;
    if (current_depth > depth) {
      return;
    }
    const auto parent_iterator = selected.find(parent_path);
    if (parent_iterator == selected.end()) {
      return;
    }
    const SelectedNode& parent = parent_iterator->second;
    const int level_width = current_depth == 1 ? width : branch_width;
    auto distribution = candidate_distribution(
        parent.token,
        parent.previous_token,
        parent.previous_previous_token,
        level_width,
        unigram_rows,
        unigram_scores,
        context_rows,
        context_scores,
        trigram_rows,
        trigram_scores);
    const double hit_mass = hit_masses[std::min<std::size_t>(
        current_depth - 1, hit_masses.size() - 1)];
    std::set<std::int64_t> seen_candidates;
    for (std::size_t index = 0; index < distribution.candidates.size();
         ++index) {
      const auto candidate = distribution.candidates[index];
      if (!seen_candidates.insert(candidate).second) {
        continue;
      }
      const Edge edge{parent.token, candidate};
      if (candidate == blocked_token_id ||
          parent.path_edges.find(edge) != parent.path_edges.end()) {
        continue;
      }
      PendingNode next;
      next.negative_score = -(
          parent_score * hit_mass * distribution.probabilities[index]);
      next.rank_path = parent_path;
      next.rank_path.push_back(static_cast<int>(index) + 1);
      next.candidate = candidate;
      next.parent_path = parent_path;
      next.path_edges = parent.path_edges;
      next.path_edges.insert(edge);
      pending.push(std::move(next));
    }
  };

  push_children(Path{}, 1.0);
  while (!pending.empty() &&
         selected_nodes.size() < static_cast<std::size_t>(node_budget)) {
    PendingNode next = pending.top();
    pending.pop();
    const auto parent_iterator = selected.find(next.parent_path);
    if (parent_iterator == selected.end()) {
      continue;
    }
    const SelectedNode& parent = parent_iterator->second;

    SelectedNode node;
    node.token = next.candidate;
    node.depth = static_cast<int>(next.rank_path.size());
    node.rank = next.rank_path.back();
    node.rank_path = next.rank_path;
    node.parent_path = next.parent_path;
    node.previous_token = parent.token;
    node.previous_previous_token = parent.previous_token;
    node.previous_previous_previous_token =
        parent.previous_previous_token;
    node.path_edges = std::move(next.path_edges);
    selected.emplace(node.rank_path, node);
    selected_nodes.push_back(std::move(node));
    push_children(next.rank_path, -next.negative_score);
  }

  std::sort(
      selected_nodes.begin(), selected_nodes.end(),
      [](const SelectedNode& lhs, const SelectedNode& rhs) {
        return lhs.rank_path < rhs.rank_path;
      });
  std::map<Path, int> flat_index_by_path;
  for (std::size_t index = 0; index < selected_nodes.size(); ++index) {
    flat_index_by_path[selected_nodes[index].rank_path] =
        static_cast<int>(index) + 1;
  }

  std::vector<std::array<std::int64_t, 4>> result;
  result.reserve(selected_nodes.size());
  for (const SelectedNode& node : selected_nodes) {
    int parent_index = 0;
    const auto parent_iterator = flat_index_by_path.find(node.parent_path);
    if (parent_iterator != flat_index_by_path.end()) {
      parent_index = parent_iterator->second;
    }
    result.push_back({
        node.token,
        static_cast<std::int64_t>(parent_index),
        static_cast<std::int64_t>(node.depth),
        static_cast<std::int64_t>(node.rank)});
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "build_score_priority_nodes",
      &build_score_priority_nodes,
      "Build a score-priority GWTR node list on the CPU",
      py::arg("root_token"),
      py::arg("root_previous_token"),
      py::arg("root_previous_previous_token"),
      py::arg("width"),
      py::arg("branch_width"),
      py::arg("depth"),
      py::arg("node_budget"),
      py::arg("blocked_token_id"),
      py::arg("unigram_rows"),
      py::arg("unigram_scores"),
      py::arg("context_rows"),
      py::arg("context_scores"),
      py::arg("trigram_rows"),
      py::arg("trigram_scores"),
      py::arg("hit_masses"));
}

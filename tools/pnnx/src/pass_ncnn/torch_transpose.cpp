// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_transpose : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.transpose         op_0        1 1 input out dim0=%dim0 dim1=%dim1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Permute";
    }

    const char* name_str() const
    {
        return "transpose";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = 0; // Default permutation type (no permutation)

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        // Get the input rank
        int input_rank = static_cast<int>(op->inputs[0]->shape.size());

        // Handle negative dimensions
        if (dim0 < 0)
        {
            dim0 += input_rank;
        }
        if (dim1 < 0)
        {
            dim1 += input_rank;
        }

        // Check if the transpose involves the batch dimension
        if (batch_index >= 0 && (dim0 == batch_index || dim1 == batch_index))
        {
            fprintf(stderr, "Error: Transpose across batch dimension is not supported!\n");
            return;
        }

        // Adjust input rank if batch dimension is present
        if (batch_index >= 0 && batch_index < input_rank)
        {
            if (dim0 > batch_index)
                dim0 -= 1;
            if (dim1 > batch_index)
                dim1 -= 1;
            input_rank -= 1;
        }

        if (input_rank > 5)
        {
            fprintf(stderr, "Error: Transpose of %d-rank tensor is not supported!\n", input_rank);
            return;
        }

        // Generate the permute order
        std::vector<int> perm(input_rank);
        for (int i = 0; i < input_rank; ++i)
        {
            perm[i] = i;
        }

        // Swap the two dimensions
        std::swap(perm[dim0], perm[dim1]);

        // Map the permute order to the parameter
        int permute_type = get_permute_type(perm, input_rank);

        if (permute_type < 0)
        {
            fprintf(stderr, "Error: Unsupported transpose permutation: ");
            for (size_t i = 0; i < perm.size(); ++i)
            {
                fprintf(stderr, "%d ", perm[i]);
            }
            fprintf(stderr, "\n");
            return;
        }

        if (permute_type == 0)
        {
            // No permutation needed
            op->type = "Noop";
        }
        else
        {
            op->params["0"] = permute_type;
        }
    }

private:
    int get_permute_type(const std::vector<int>& perm, int input_rank) const
    {
        // This function maps the permutation vector to the corresponding parameter value
        // in the NCNN Permute layer. For input_rank up to 5, we define mappings here.

        // For 1D tensor, only identity permutation exists
        if (input_rank == 1)
        {
            if (perm == std::vector<int>{0})
                return 0;
            else
                return -1;
        }

        // For 2D tensors
        if (input_rank == 2)
        {
            if (perm == std::vector<int>{0, 1})
                return 0;
            else if (perm == std::vector<int>{1, 0})
                return 1;
            else
                return -1;
        }

        // For 3D tensors
        if (input_rank == 3)
        {
            static const std::map<std::vector<int>, int> permute_map_3d = {
                {{0, 1, 2}, 0},
                {{0, 2, 1}, 1},
                {{1, 0, 2}, 2},
                {{1, 2, 0}, 3},
                {{2, 0, 1}, 4},
                {{2, 1, 0}, 5},
            };

            auto it = permute_map_3d.find(perm);
            if (it != permute_map_3d.end())
                return it->second;
            else
                return -1;
        }

        // For 4D tensors
        if (input_rank == 4)
        {
            static const std::map<std::vector<int>, int> permute_map_4d = {
                {{0, 1, 2, 3}, 0},
                {{0, 1, 3, 2}, 1},
                {{0, 2, 1, 3}, 2},
                {{0, 2, 3, 1}, 3},
                {{0, 3, 1, 2}, 4},
                {{0, 3, 2, 1}, 5},
                {{1, 0, 2, 3}, 6},
                {{1, 0, 3, 2}, 7},
                {{1, 2, 0, 3}, 8},
                {{1, 2, 3, 0}, 9},
                {{1, 3, 0, 2}, 10},
                {{1, 3, 2, 0}, 11},
                {{2, 0, 1, 3}, 12},
                {{2, 0, 3, 1}, 13},
                {{2, 1, 0, 3}, 14},
                {{2, 1, 3, 0}, 15},
                {{2, 3, 0, 1}, 16},
                {{2, 3, 1, 0}, 17},
                {{3, 0, 1, 2}, 18},
                {{3, 0, 2, 1}, 19},
                {{3, 1, 0, 2}, 20},
                {{3, 1, 2, 0}, 21},
                {{3, 2, 0, 1}, 22},
                {{3, 2, 1, 0}, 23},
            };

            auto it = permute_map_4d.find(perm);
            if (it != permute_map_4d.end())
                return it->second;
            else
                return -1;
        }

        // For 5D tensors
        if (input_rank == 5)
        {
            static const std::map<std::vector<int>, int> permute_map_5d = generate_5d_permute_map();

            auto it = permute_map_5d.find(perm);
            if (it != permute_map_5d.end())
                return it->second;
            else
                return -1;
        }

        // Unsupported input rank
        return -1;
    }

    std::map<std::vector<int>, int> generate_5d_permute_map() const
    {
        // Generate mapping for all possible single swaps (transpositions) in 5D tensors
        std::map<std::vector<int>, int> permute_map;
        int param_value = 0;

        // Identity permutation
        permute_map[{0, 1, 2, 3, 4}] = param_value++;

        // All possible single swaps
        for (int i = 0; i < 5; ++i)
        {
            for (int j = i + 1; j < 5; ++j)
            {
                std::vector<int> perm = {0, 1, 2, 3, 4};
                std::swap(perm[i], perm[j]);
                permute_map[perm] = param_value++;
            }
        }

        return permute_map;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_transpose, 20)

} // namespace ncnn

} // namespace pnnx
// Tensor_permute.cpp

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C)
// 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

class Tensor_permute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const override
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const override
    {
        return "Permute";
    }

    const char* name_str() const override
    {
        return "permute";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const override
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        const std::vector<int>& dims = captured_params.at("dims").ai;

        int input_rank = static_cast<int>(op->inputs[0]->shape.size());

        if (input_rank == 0)
        {
            // Assume input is fine
            input_rank = static_cast<int>(dims.size());
        }

        if (batch_index >= 0 && batch_index < input_rank)
            input_rank -= 1;

        if (input_rank > 5)
        {
            fprintf(stderr, "Error: permute %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        // Drop permute batch index
        std::vector<int> new_dims;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (dims[i] == batch_index)
                continue;

            int new_dim = dims[i] > batch_index ? dims[i] - 1 : dims[i];
            new_dims.push_back(new_dim);
        }

        if (input_rank != static_cast<int>(new_dims.size()))
        {
            fprintf(stderr, "Error: permute %d-rank tensor with %d-rank dims is not possible\n", input_rank, static_cast<int>(new_dims.size()));
            return;
        }

        // Validate the permutation
        std::vector<int> sorted_dims = new_dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        for (int i = 0; i < input_rank; ++i)
        {
            if (sorted_dims[i] != i)
            {
                fprintf(stderr, "Error: Invalid permutation dimensions\n");
                return;
            }
        }

        // Check if permutation is identity
        bool is_identity = true;
        for (int i = 0; i < input_rank; ++i)
        {
            if (new_dims[i] != i)
            {
                is_identity = false;
                break;
            }
        }

        if (is_identity)
        {
            // No permutation needed
            op->type = "Noop";
            return;
        }

        // Set order_type to -1 to indicate custom permutation
        op->params["0"] = -1;

        // Set the permutation parameters
        for (int i = 0; i < input_rank; ++i)
        {
            std::string key = std::to_string(i + 1); // Parameter keys start from "1"
            op->params[key] = new_dims[i];
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_permute, 20)

} // namespace ncnn

} // namespace pnnx

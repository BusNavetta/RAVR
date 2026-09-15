using DelimitedFiles

if length(ARGS) != 2
    error("usage: qaq_official_smoke.jl OFFICIAL_REPO IO_DIRECTORY")
end
official_repo = abspath(ARGS[1])
io_directory = abspath(ARGS[2])
include(joinpath(official_repo, "src", "model", "saq_v2.jl"))

meta = vec(Int.(readdlm(joinpath(io_directory, "meta.csv"), ',', Int)))
n, d, m, k, kv, iterations = meta
ds = div(d, m)
points = Matrix{Float64}(readdlm(joinpath(io_directory, "points.csv"), ','))
labels = vec(Int.(readdlm(joinpath(io_directory, "labels.csv"), ',', Int)))
codebook_rows = Matrix{Float64}(readdlm(joinpath(io_directory, "codebook.csv"), ','))
matrix_rows = Matrix{Float64}(readdlm(joinpath(io_directory, "matrices.csv"), ','))

codebook = zeros(Float64, m, k, ds)
for stage in 1:m, symbol in 1:k
    codebook[stage, symbol, :] = codebook_rows[(stage - 1) * k + symbol, :]
end
matrices = zeros(Float64, kv, d, d)
for cluster in 1:kv
    matrices[cluster, :, :] = matrix_rows[(cluster - 1) * d + 1:cluster * d, :]
end

data = DATA_s(points, points[1:min(2, n), :])
global saq = SAQ_class(m, k, data)
spmi = semi_pos_matrix_s(matrices, labels)
codes, loss = encode(points, codebook, spmi)
code_indices = pqcode_To_pqcode2(codes, k, d)
construct = compute_construct_ves(points, spmi)
updated_flat = update_pqcodebook(saq, construct, code_indices, spmi)
updated = pq_codebook_1dto3d(updated_flat, m, k, ds)
updated_rows = zeros(Float64, m * k, ds)
for stage in 1:m, symbol in 1:k
    updated_rows[(stage - 1) * k + symbol, :] = updated[stage, symbol, :]
end

writedlm(joinpath(io_directory, "official_codes.csv"), codes, ',')
writedlm(joinpath(io_directory, "official_loss.csv"), [loss], ',')
writedlm(joinpath(io_directory, "official_updated_codebook.csv"), updated_rows, ',')

using CirculantAttention
using Documenter

push!(LOAD_PATH,"../src/") # remove when depolyed

DocMeta.setdocmeta!(CirculantAttention, :DocTestSetup, :(using CirculantAttention); recursive=true)

makedocs(;
    modules=[CirculantAttention],
    authors="nikopj and contributors",
    repo="https://github.com/nikopj/CirculantAttention.jl/blob/{commit}{path}#{line}",
    sitename="CirculantAttention.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nikopj.github.io/CirculantAttention.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nikopj/CirculantAttention.jl",
    devbranch="main",
)

function Zygote.accum(x::T, y::T) where T <: Circulant 
    x === nothing ? y : 
    y === nothing ? x :
    x + y
end

Zygote.accum(x::T, y::T, z::T...) where T <: Circulant = Zygote.accum(Zygote.accum(x, y), z...)


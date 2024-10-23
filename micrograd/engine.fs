module Micrograd.Engine

type Value(data: float, ?_children: seq<Value>, ?_op: string) =
    let _children = defaultArg _children Seq.empty
    let _op = defaultArg _op ""
    let mutable grad = 0.0
    let mutable _backward = fun () -> ()
    let _prev = Set.ofSeq _children

    member this.Data = data
    member this.Grad
        with get() = grad
        and set(value) = grad <- value

    member this.Add(other: Value) =
        let other = if other :? Value then other else Value(other.Data)
        let out = Value(this.Data + other.Data, [this; other], "+")
        let _backward() =
            this.Grad <- this.Grad + out.Grad
            other.Grad <- other.Grad + out.Grad
        out._backward <- _backward
        out

    member this.Mul(other: Value) =
        let other = if other :? Value then other else Value(other.Data)
        let out = Value(this.Data * other.Data, [this; other], "*")
        let _backward() =
            this.Grad <- this.Grad + other.Data * out.Grad
            other.Grad <- other.Grad + this.Data * out.Grad
        out._backward <- _backward
        out

    member this.Pow(other: float) =
        let out = Value(this.Data ** other, [this], $"**{other}")
        let _backward() =
            this.Grad <- this.Grad + (other * this.Data ** (other - 1.0)) * out.Grad
        out._backward <- _backward
        out

    member this.ReLU() =
        let out = Value(if this.Data < 0.0 then 0.0 else this.Data, [this], "ReLU")
        let _backward() =
            this.Grad <- this.Grad + (if out.Data > 0.0 then 1.0 else 0.0) * out.Grad
        out._backward <- _backward
        out

    member this.Backward() =

        let topo = ResizeArray<Value>()
        let visited = HashSet<Value>()
        let rec build_topo v =
            if not (visited.Contains v) then
                visited.Add v |> ignore
                for child in v._prev do
                    build_topo child
                topo.Add v
        build_topo this
        this.Grad <- 1.0
        for v in List.rev topo do
            v._backward()

    member this.Neg() = this.Mul(Value(-1.0))
    member this.RAdd(other: Value) = this.Add(other)
    member this.Sub(other: Value) = this.Add(other.Neg())
    member this.RSub(other: Value) = other.Add(this.Neg())
    member this.RMul(other: Value) = this.Mul(other)
    member this.Div(other: Value) = this.Mul(other.Pow(-1.0))
    member this.RDiv(other: Value) = other.Mul(this.Pow(-1.0))

    override this.ToString() = $"Value(data={this.Data}, grad={this.Grad})"

    interface System.IComparable with
        member this.CompareTo(other) =
            if this.Data < other.Data then -1
            elif this.Data > other.Data then 1
            else 0

module Micrograd.NN

open System
open Micrograd.Engine

type Module() =
    abstract member ZeroGrad: unit -> unit
    abstract member Parameters: unit -> seq<Value>
    default this.ZeroGrad() =
        for p in this.Parameters() do
            p.Grad <- 0.0
    default this.Parameters() = Seq.empty

type Neuron(nin: int, ?nonlin: bool) =
    inherit Module()
    let w = [for _ in 1..nin -> Value(random.NextDouble() * 2.0 - 1.0)]
    let b = Value(0.0)
    let nonlin = defaultArg nonlin true

    member this.Call(x: seq<Value>) =
        let act = Seq.fold2 (fun acc wi xi -> acc + wi * xi) b w x
        if nonlin then act.ReLU() else act

    override this.Parameters() = seq { yield! w; yield b }

    override this.ToString() = sprintf "%sNeuron(%d)" (if nonlin then "ReLU" else "Linear") (List.length w)

type Layer(nin: int, nout: int, ?nonlin: bool) =
    inherit Module()
    let neurons = [for _ in 1..nout -> Neuron(nin, ?nonlin=nonlin)]

    member this.Call(x: seq<Value>) =
        let out = [for n in neurons -> n.Call(x)]
        if List.length out = 1 then out.Head else out

    override this.Parameters() = seq { for n in neurons do yield! n.Parameters() }

    override this.ToString() = sprintf "Layer of [%s]" (String.Join(", ", neurons |> List.map string))

type MLP(nin: int, nouts: int list) =
    inherit Module()
    let sz = nin :: nouts
    let layers = [for i in 0..List.length nouts - 1 -> Layer(sz.[i], sz.[i+1], nonlin=(i <> List.length nouts - 1))]

    member this.Call(x: seq<Value>) =
        let mutable x = x
        for layer in layers do
            x <- layer.Call(x)
        x

    override this.Parameters() = seq { for layer in layers do yield! layer.Parameters() }

    override this.ToString() = sprintf "MLP of [%s]" (String.Join(", ", layers |> List.map string))

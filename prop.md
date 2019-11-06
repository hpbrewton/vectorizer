# Illustrative Example

As an example, 
consider the following problem that we want to create.
We want a model from molecule to the spectrum it maps to.

We can write this function stub in Rust as follows:
```rust
fn findSpectra(molecule: Molecule) -> Spectra
```
You might already be able to see where this going:
we have a set of pairs of molecules and their observed spectra,
and we want to find model learned using traditional ML techniques that implements this function.
In short we do synthesis using ML.

The major problem with this approach is that `Molecule` and `Spectra` have the below definitions:
```rust
struct Molecule {
    atomicNumbers: Vec<u8>,
    bonds: Vec<(u64, u64, u8)> // atom a, atom b, and the strength of the bond
}

type Spectra = Vec<f64> // mass/charge spectra at some scale, but could be different sizes
```
As we can see both of these structures have variable sizes, 
and so we can not just say convert the input and the output into vectors and Bob's your uncle.

One approach to take in solving this problem is to use custom kernel methods that take structures
and sample form them.
For example,
a common approach in converting molecules to vectors is doing a graph kernel:
essentially we just take a random walk on the graph such that we get a constant size sub graph.
We can do a different approach with the spectra wherein we sample from the spectra at even distances 
to create a constant $n$ sized vector.
There are many such methods to convert variable sized structures 
into vectorizable objects, 
and this just one example.
With this one example we could use the vectorizing methods to create a model to learn 
the aforementioned `findSpectra` method.

# Formal definition of the problem

Given an input type $\mathcal{I}$ and an output type $\mathcal{O}$,
we want to create two coding functions.
These coding functions mapped values of some type that may not be fixed size
to a fixed size vector over some alphabet.

$$ C_i : \mathcal{I} \rightarrow \Sigma^n \quad C_o : \mathcal{O} \rightarrow \Sigma^m $$

Unlike the traditional compression coding problem where want the coding function to minimize the size of the output of the coding functions,
we instead want to be able to just have a fixed size.

# Kernel of an idea

Essentially we want to create a mapping from some type system to a vectorization.
Some more ideas on these mappings are listed below:
- Fixed size integers can just be directly mapped to their bit vector
- The vectorization of the product of types is just the concatenation of the vectorization of individual types.
- We can use dependent typing information within a struct to realize that we need to sample the same elements of node information with their respective adjacency value.
- We might be able to use interface information to gather more information
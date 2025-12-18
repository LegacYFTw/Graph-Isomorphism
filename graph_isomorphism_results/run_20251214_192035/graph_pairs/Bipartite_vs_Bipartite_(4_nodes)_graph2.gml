graph [
  name "complete_bipartite_graph(2, 2)"
  node [
    id 0
    label "1"
    bipartite 0
  ]
  node [
    id 1
    label "2"
    bipartite 0
  ]
  node [
    id 2
    label "3"
    bipartite 1
  ]
  node [
    id 3
    label "0"
    bipartite 1
  ]
  edge [
    source 0
    target 2
  ]
  edge [
    source 0
    target 3
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 1
    target 3
  ]
]

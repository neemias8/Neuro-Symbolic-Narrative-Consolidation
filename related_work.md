# Related Work

## Multi-Document Information Consolidation
Information consolidation goes beyond traditional Multi-Document Summarization (MDS). While MDS focuses on compression and redundancy removal (Lebanoff et al., 2018; Fabbri et al., 2019), consolidation aims to synthesize scattered information into a coherent whole without loss of detail. Recent works in MDS, such as PRIMERA (Xiao et al., 2022), use pre-training objectives tailored for multi-document inputs but often struggle with maintaining long-range chronological coherence in narrative domains. Our work aligns with the definition proposed in the Dagstuhl Seminar 19452 (2019), emphasizing the reconstruction of a unified narrative from fragmented sources.

## Sentence Ordering and Coherence
Sentence ordering is a fundamental task in text generation (Barzilay and Lapata, 2008). Neural approaches typically treat it as a ranking or pointer network problem (Gong et al., 2016; Cui et al., 2018). However, these methods are often applied as post-processing steps or on short texts. The Topological-Temporal Alignment Graph (TAEG) differs by enforcing structural constraints *before* generation, acting as a neuro-symbolic guide rather than a post-hoc reordering mechanism. This ensures that the generated narrative adheres to a verified chronological backbone.

## Neuro-Symbolic and Knowledge-Grounded Generation
Our approach falls under the umbrella of neuro-symbolic AI, where symbolic knowledge (the TAEG structure) guides neural generation (LLMs). This is similar to Knowledge-Grounded Generation (KGG) tasks (Zhao et al., 2020), where external knowledge graphs enhance consistency. Unlike standard KGG which uses general knowledge bases (e.g., Wikidata), our method constructs a dynamic, document-specific temporal graph to ground the generation process, mitigating hallucinations and chronological inconsistencies common in pure neural models.

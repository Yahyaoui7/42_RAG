# RAG Against the Machine — Architecture Diagrams as Code

> All diagrams are Mermaid. Paste any block into https://mermaid.live to render/export.

---

## 1. System Architecture — Main Pipeline



```mermaid
graph TD
    A["Raw Repository<br/>data/raw - vLLM codebase"] --> B["scanner.py<br/>scan_files - Path.rglob"]

    B --> C["chunking/python_chunker.py<br/>chunk_python_file - AST"]
    B --> D["chunking/markdown_chunker.py<br/>chunk_markdown_file - headers"]

    C --> E["chunking/base.py<br/>Chunk dataclass + validate + compat guard"]
    D --> E

    E --> F["tokenizer.py<br/>tokenize - regex lower alphanum"]
    F --> G["indexer.py<br/>LexicalIndexer - BM25Okapi rank_bm25"]

    G --> H[("data/processed/index.pkl<br/>IndexData: bm25 + chunks + tokenized_corpus")]

    H --> I["retriever.py<br/>search - get_scores + argsort top-K"]
    J["User Query<br/>str"] --> I

    I --> K["models.py<br/>StudentSearchResults"]

    K --> L["prompts.py<br/>build_prompt - context + question"]
    J --> L

    L --> M["generator.py<br/>Qwen3-0.6B - transformers AutoModel"]
    M --> N["models.py<br/>StudentSearchResultsAndAnswer"]

    K --> O["evaluator.py<br/>recall@k - IoU > 0.05 + file_path verbatim"]
    P["Ground Truth<br/>data/datasets/AnsweredQuestions"] --> O

    O --> Q["Recall@K Score<br/>@1 @3 @5 @10 - target 80% docs / 50% code"]
```


## 2. Evaluation Flow (Isolated)

```mermaid id="r3n7qx"
graph LR
    S["StudentSearchResults<br/>data/output/search_results"] --> EVAL["evaluator.py<br/>is_match: file_path == AND IoU &gt; 0.05"]
    T["Ground Truth<br/>RagDataset AnsweredQuestions"] --> EVAL

    EVAL --> R["Recall@K<br/>found / total_sources"]

    R --> OK{"Recall@5 &gt;= 80% docs<br/>50% code?"}

    OK -->|No| FIX["Fix chunking / BM25 params<br/>re-index -> re-search -> re-eval"]
    OK -->|Yes| NEXT["Phase 4 - Generation"]

    FIX --> S
```

## 3. Retrieval vs Generation Separation (Key Principle)

```mermaid
graph TD
    subgraph RETRIEVAL["RETRIEVAL - test independently first"]
        R1["Question"] --> R2["BM25 Index"] --> R3["Top-K Chunks"] --> R4["StudentSearchResults JSON"]
    end

    subgraph GENERATION["GENERATION - layered on top"]
        G1["Question + Top-K Chunks"] --> G2["Prompt Builder"] --> G3["Qwen3-0.6B"] --> G4["StudentSearchResultsAndAnswer JSON"]
    end

    R4 -.-> G1
```

## 4. File Tree -> Architecture Component Mapping

```mermaid
graph TD
    subgraph SRC [src/ - python -m src]
        A1[__main__.py<br/>CLI - Fire]
        A2[config.py<br/>constants + paths]
        A3[models.py<br/>Pydantic VI.4]
        A4[scanner.py<br/>File Scanner]
        A5[tokenizer.py<br/>Tokenizer]
        A6[indexer.py<br/>LexicalIndexer]
        A7[retriever.py<br/>Retriever]
        A8[prompts.py<br/>Prompt Builder]
        A9[generator.py<br/>LLM Loader]
        A10[evaluator.py<br/>Moulinette clone]
        subgraph CHUNKING [chunking/]
            C1[base.py<br/>Chunk + validate + chunk_to_minimal_source<br/>split_oversized_chunk is compat guard]
            C2[python_chunker.py<br/>enforces max inline line-aware]
            C3[markdown_chunker.py<br/>enforces max inline paragraph-aware]
        end
    end
    subgraph DATA [data/]
        D1[data/raw/<br/>corpus - FIXED for index]
        D2[data/processed/<br/>index.pkl - FIXED for index]
        D3[data/datasets/<br/>Unanswered + Answered]
        D4[data/output/search_results/&lt;Scope&gt;]
        D5[data/output/search_results_and_answer/&lt;Scope&gt;]
    end
    A1 -. index .-> D1
    A1 -. index .-> D2
    A1 -. search_dataset .-> D3
    A1 -. search_dataset .-> D4
    A1 -. answer_dataset .-> D5
```

## 5. Detailed Class / Module Diagram — Variables + Functions per File

```mermaid
classDiagram
    class config_py {
        <<module>> config.py
        +Path RAW_DIR
        +Path PROCESSED_DIR
        +Path INDEX_PATH
        +int DEFAULT_MAX_CHUNK_SIZE = 2000
        +float BM25_K1 = 1.5
        +float BM25_B = 0.75
        +str QWEN_MODEL_ID
        +int QWEN_MAX_NEW_TOKENS = 512
    }
    class Chunk {
        <<dataclass frozen>> chunking/base.py
        +str text
        +str file_path
        +int first_character_index
        +int last_character_index
        +split_oversized_chunk(Chunk, int) List~Chunk~  // compat guard: raises if >max
        +validate_chunk(Chunk, str) bool
        +chunk_to_minimal_source(Chunk) MinimalSource
    }
    class python_chunker_py {
        <<module>> chunking/python_chunker.py
        +chunk_python_file(Path, int) List~Chunk~
        -_build_line_offsets(str) List~int~
        -_lineno_to_char_index(List~int~, int, int) int
        -_get_node_span(node, List~int~, str) Tuple~int,int~
    }
    class markdown_chunker_py {
        <<module>> chunking/markdown_chunker.py
        +chunk_markdown_file(Path, int) List~Chunk~
        -_split_by_headers(str) List of Tuple
        -_split_by_paragraphs(str) List of Tuple
    }
    class tokenizer_py {
        <<module>> tokenizer.py
        +Pattern _TOKEN_RE
        +tokenize(str) List~str~
    }
    class IndexData {
        <<dataclass>> indexer.py
        +BM25Okapi bm25
        +List~Chunk~ chunks
        +List of List of str tokenized_corpus
        +build_index(List~Chunk~) IndexData
        +save_index(IndexData, Path) void
        +load_index(Path) IndexData
        +build_and_save_from_raw(int) void
    }
    class retriever_py {
        <<module>> retriever.py
        +search(str query, int k, IndexData) List~MinimalSource~
        +search_one(str, Path, int) List~MinimalSource~
        +format_as_student_results(List~UnansweredQuestion~, int, IndexData) StudentSearchResults
    }
    class prompts_py {
        <<module>> prompts.py
        +str SYSTEM_PROMPT
        +str PROMPT_TEMPLATE
        +build_prompt(str question, List~str~ contexts) str
        +load_chunk_text(MinimalSource) str
        +truncate_contexts(List~str~, int budget) List~str~
    }
    class generator_py {
        <<module>> generator.py
        -AutoModel _model = None
        -AutoTokenizer _tokenizer = None
        +get_model_and_tokenizer() Tuple~Tokenizer, Model~
        +generate_answer(str prompt, int max_new_tokens, float temp) str
        +generate_batch(List~str~ prompts) List~str~
    }
    class evaluator_py {
        <<module>> evaluator.py
        +compute_iou(int a0,int a1,int b0,int b1) float
        +is_match(MinimalSource retr, MinimalSource truth) bool
        +recall_at_k(StudentSearchResults, RagDataset, int k) Dict~int,float~
        +evaluate_files(Path student, Path truth) void
    }
    class CLI {
        <<Fire CLI>> __main__.py
        +index(max_chunk_size: int = 2000) void
        +search(query: str, k: int = 5) void
        +search_dataset(dataset_path: str, k: int, save_directory: str) void
        +answer(query: str, k: int = 5) void
        +answer_dataset(student_search_results_path: str, save_directory: str) void
        +evaluate(student_search_results_path: str, dataset_path: str) void
        +main() void
    }
    class Makefile {
        <<Makefile>>
        +install: uv sync
        +run: uv run python -m src $(ARGS)
        +debug: uv run python -m pdb -m src $(ARGS)
        +clean: rm __pycache__ .mypy_cache
        +lint: flake8 + mypy
        +lint-strict: flake8 + mypy --strict
    }

    python_chunker_py ..> Chunk : creates
    markdown_chunker_py ..> Chunk : creates
    IndexData ..> Chunk : stores
    IndexData ..> tokenizer_py : uses tokenize
    retriever_py ..> IndexData : loads
    retriever_py ..> tokenizer_py : uses tokenize
    CLI ..> IndexData : calls
    CLI ..> retriever_py : calls
    CLI ..> generator_py : calls
    CLI ..> evaluator_py : calls
    prompts_py ..> generator_py : feeds prompt
    CLI ..> prompts_py : calls
    Makefile ..> CLI : invokes
```

## 6. Pydantic Data Models (PDF VI.4 — exact, do not rename)

```mermaid
classDiagram
    class MinimalSource {
        <<BaseModel>>
        +str file_path
        +int first_character_index
        +int last_character_index
    }
    class UnansweredQuestion {
        <<BaseModel>>
        +str question_id = uuid4()
        +str question
    }
    class AnsweredQuestion {
        <<BaseModel>>
        +List~MinimalSource~ sources
        +str answer
    }
    class RagDataset {
        <<BaseModel>>
        +List of RagQuestion rag_questions
    }
    class MinimalSearchResults {
        <<BaseModel>>
        +str question_id
        +str question
        +List~MinimalSource~ retrieved_sources
    }
    class MinimalAnswer {
        <<BaseModel>>
        +str answer
    }
    class StudentSearchResults {
        <<BaseModel>>
        +List~MinimalSearchResults~ search_results
        +int k
    }
    class StudentSearchResultsAndAnswer {
        <<BaseModel>>
        +List~MinimalAnswer~ search_results
        +int k
    }

    UnansweredQuestion <|-- AnsweredQuestion : inherits
    MinimalSearchResults <|-- MinimalAnswer : inherits
    MinimalSearchResults *-- MinimalSource : retrieved_sources
    AnsweredQuestion *-- MinimalSource : sources
    RagDataset o-- AnsweredQuestion : rag_questions
    RagDataset o-- UnansweredQuestion : rag_questions
    StudentSearchResults o-- MinimalSearchResults : search_results
    StudentSearchResultsAndAnswer o-- MinimalAnswer : search_results
```

## 7. Sequence — Indexing

```mermaid

sequenceDiagram
    participant CLI as Main CLI
    participant S as Scanner
    participant PC as Python Chunker
    participant MC as Markdown Chunker
    participant T as Tokenizer
    participant I as Indexer

    CLI->>S: scan_files(RAW_DIR)
    S-->>CLI: Sorted Python and Markdown file paths

    loop For each file
        alt Python file
            CLI->>PC: chunk_python_file(path, max_chunk_size)
            PC->>PC: Parse AST and find classes/functions
            PC->>PC: Split oversized code units
            PC-->>CLI: List of Chunk objects
        else Markdown file
            CLI->>MC: chunk_markdown_file(path, max_chunk_size)
            MC->>MC: Split by headings and paragraphs
            MC->>MC: Split oversized paragraphs
            MC-->>CLI: List of Chunk objects
        end
    end

    Note over PC,MC: Chunk objects store text, file path, and character positions

    CLI->>T: Tokenize each chunk's text
    T-->>CLI: List of token lists

    CLI->>I: build_index(chunks)
    Note over I: BM25Okapi with k1=1.5 and b=0.75
    I-->>CLI: IndexData

    CLI->>I: save_index(IndexData, data/processed/index.pkl)
    I-->>CLI: Index saved
```

## 8. Sequence — Retrieval (search / search_dataset)

```mermaid
sequenceDiagram
    participant CLI as search_dataset()
    participant L as indexer.py:load_index()
    participant R as retriever.py:search()
    participant T as tokenizer.py
    participant M as models.py
    CLI->>L: load_index(data/processed/index.pkl)
    L-->>CLI: IndexData(bm25, chunks)
    CLI->>CLI: load RagDataset from --dataset_path
    loop for each UnansweredQuestion with tqdm
        CLI->>T: tokenize(question)
        T-->>R: List[str] tokens
        R->>R: bm25.get_scores(tokens) -> np.argsort()[::-1][:k]
        alt k==0 or empty query
            R-->>CLI: [] no crash
        else
            R-->>CLI: List[MinimalSource] file_path verbatim
        end
    end
    CLI->>M: StudentSearchResults(search_results, k).model_dump_json()
    CLI->>CLI: write to --save_directory/<dataset_filename>.json
```

## 9. Sequence — Generation (answer / answer_dataset)

```mermaid
sequenceDiagram
    participant CLI as answer_dataset()
    participant R as retriever.py
    participant P as prompts.py
    participant G as generator.py
    participant M as models.py
    CLI->>CLI: load StudentSearchResults from --student_search_results_path
    CLI->>G: get_model_and_tokenizer() lazy singleton
    G-->>CLI: (tokenizer, model) Qwen/Qwen3-0.6B
    loop for each MinimalSearchResults with tqdm
        CLI->>P: load_chunk_text(src) for each retrieved_sources
        P-->>CLI: List[str] contexts
        CLI->>P: build_prompt(question, contexts)
        P->>P: truncate if > token budget
        P-->>G: prompt str
        G->>G: tokenizer(prompt) -> model.generate(max_new_tokens=512)
        G-->>CLI: answer text decode
        CLI->>M: MinimalAnswer(question_id, question, retrieved_sources, answer)
    end
    CLI->>M: StudentSearchResultsAndAnswer(search_results, k).model_dump_json()
    CLI->>CLI: write to --save_directory/<dataset_filename>.json
```

## 10. Sequence — Evaluation (your own iterate)

```mermaid
sequenceDiagram
    participant E as evaluator.py:evaluate()
    participant S as StudentSearchResults
    participant T as RagDataset truth
    E->>S: load student JSON
    E->>T: load truth JSON
    loop for each question
        loop for each truth source
            E->>E: is_match(ret, truth) => file_path == AND IoU>0.05
            Note over E: IoU = intersection / union<br/>intersection = max(0, min(a1,b1)-max(a0,b0))<br/>union = max(a1,b1)-min(a0,b0)
        end
        E->>E: recall = found / len(truth.sources)
    end
    E-->>CLI: print Recall@1 @3 @5 @10 table
    Note over E,CLI: Official is ./moulinette evaluate_student_search_results student.json truth.json --k 10 --max_context_length 2000
```

## 11. CLI Command Map

```mermaid
graph LR
    subgraph FIRE [fire.Fire CLI - src/__main__.py]
        C1[index<br/>--max_chunk_size int=2000<br/>FIXED data/raw -> data/processed]
        C2[search<br/>query pos + --k int<br/>stdout JSON]
        C3[search_dataset<br/>--dataset_path --k --save_directory<br/>tqdm + never crash]
        C4[answer<br/>query pos + --k int<br/>Qwen + prompt]
        C5[answer_dataset<br/>--student_search_results_path --save_directory<br/>tqdm]
        C6[evaluate<br/>--student_search_results_path --dataset_path<br/>local recall@k]
    end
```

## 12. Implementation Order (Dependency Graph)

```mermaid id="m8p4kx"
graph TD
    P1["Phase 1<br/>models.py + config.py + scanner.py"] --> P2["Phase 2<br/>chunking/base + python + markdown + tokenizer.py"]
    P2 --> P3["Phase 2 cont<br/>indexer.py + index CLI<br/>&lt;5 min check"]
    P3 --> P4["Phase 3<br/>retriever.py + search + search_dataset"]
    P4 --> P5["Phase 3 cont<br/>evaluator.py<br/>80% docs 50% code @5"]
    P5 --> P6["Phase 4<br/>prompts.py + generator.py<br/>answer + answer_dataset"]
    P6 --> P7["Phase 5<br/>types X|None + flake8 + mypy --strict<br/>edge cases + tqdm + README"]
```

# Overview

이 레포에는 executorch오픈소스의 분석과 pytorch, executorch의 실행시간 및 오차를 구하는 프로그램을 구현했습니다.

## Program Overview

- language: python 3.13
- server framework: fastapi
- database: sqlite3
- orm library: sqlmodel

### 파일 구조
```bash
.
├── api-test    # curl files for testing API in local
├── data    # generated data when application run for sqlite3 file
│   └── database.db     # sqlite3 database
├── docker-compose.yml
├── Dockerfile
├── engine
│   └── task_engine.py  # task engine entry point for async tasks
├── main.py     # API server entry point
├── models      # sqlmodel models
│   ├── model.py
│   └── task.py
├── requirements.txt    # package dependencies
├── saved_models    # generated directory when application run and model is uploded
└── task
    ├── benchmark.py    # logic to compare pytorch and executorch model
    ├── executor.py     # simple controller to run benchmark
    └── tests
        ├── test_benchmark.py
        ├── test_executor.py
        └── test.pt2
```

### 파일 실행 방법

```bash
docker compose up
```

API 호출은 api-test에 있는 파일들을 사용하시거나, localhost:8000/docs에 접근하시면 swagger사용이 가능합니다.

아래는 테스트 실행 방법
```bash
pytest
```

### 설계

비동기 큐는 sqlite3를 사용한 rdb큐를 사용했습니다. 프러덕션 환경에서는 mysql, postgresql를 사용하면 확장성 있게 비동기 작업을 처리할 수 있습니다.
요청의 빈도수가 높고 병렬적으로 빠르게 처리가 되어야한다면 redis, kafka와 같은 큐를 사이에 두겠지만, 모델 실행비교는 rdb로도 충분히 비동기 실행을 할 수 있을 것 같다는 판단이 있었습니다.


## Executorch Analysis

### Overview

Executorch는 pytorch 모델을 온디바이스에 최적화된 모델로 실행할 수 있게 해주는 컴파일 & 런타임 레이어입니다.

Executorch는 이를 위해 크게 컴파일과 런타임 레이어를 제공합니다. 우선 컴파일 레이어는 Pytorch에서 export한 ExportProgram을 기반으로 Edge Dialect, Backend Dialect, 그리고 Memory planning 순서를 통해 pte파일로 lowering을 진행합니다.
lowering이라고 불리는 이유는 python언어로 아무 서버에서나 돌릴 수 있었던 모델이 점점 하드웨어에 최적화된 로우레밸로 내려가기 때문입니다.

Edge dialect와 Backend dialect는 각각 ExportProgram을 좀 더 lowering한 것이지만, Backend dialect는 좀 더 하드웨어에 최적화된 형태의 모델로 lowering한 것입니다. Backend dialect는 특정 하드웨어에서 실행할 수 있는 operator를 delegate해서 실행하기 때문에 target aware한 최적화를 한 것입니다. 그리고 quantization도 컴파일 레이어에서 진행합니다. executorch는 아직 PTQ만 지원하고 QAT는 정식으로는 지원하지 않는 것 같습니다.

런타임 레이어는 컴파일 레이어에서 경량화 및 최적화가 진행된 pte모델을 실제 디바이스에서 실행하여 모델 추론에 사용할 수 있게 해주는 레이어입니다.
경량화된 모델은 모델이 어떤 함수들로 실행이 되어야하는지 graph형태로 정의되어 있다고 이해했습니다. 그렇다면 이 함수들이 실제 디바이스에서 무엇에 의해 실행되어야 하는지 맵핑해주는 것이 존재해야할텐데, 그것이 Registry입니다.
Backend를 예로 들면, 특정 하드웨어에서 실행이 되는 기능들은 Backend Registry를 통해 맵핑이 되어 delegation이 되는 그림인 것 같습니다.


### Terms

- Backend: 특정 하드웨어 (GPU, NPU)나 소프트웨어 스택 (XNNPACK)을 의미하고 최적화된 모델의 특정 부분을 실행시켜 모델 실행시 높은 성능과 효율을 낼 수 있게 해줍니다.
- Delegation: 최적화된 모델의 특정 부분을 Backend에 위임해 실행시키는 것입니다.
- ExportedProgram: torch.export의 결과물로 파라미터와 가중치를 가지는 연산 가능한 노드의 그래프형태입니다.
- Lowering: 모델을 하드웨어 최적화된 상태로 변환하는 과정입니다.
- Partitioner: 모델에서 Backend에 delegate할 파트를 말 그대로 partition하는 컴포넌트입니다.
- Quantization: 모델을 경량화하는 기법입니다. 보통 모델 가중치 data type을 손실이 적지만 크기가 작은 형태로 변경해서 구현합니다. fp32 -> int8

### 지원 범위 및 한계점

다양한 Backend를 지원하고 있고 Quantization, Memory planning등의 최적화도 가능합니다.
Backend 지원이 되어 있지 않다면, 이것을 직접 구현해야하는 한계가 있는 것 같습니다.
또한 최적화된 모델이 추론으로 사용될수는 있지만, 학습을 위해서는 사용될 수 없다는 한계가 있는 것 같습니다.

### 소스코드 구조 분석 내용

아래는 executorch가 제공하는 소스파일 구조를 간략화한 것입니다.
```
executorch
├── backends - Backend delegate implementations for various hardware targets. Each backend uses partitioner to split the graph into subgraphs that can be executed on specific hardware, quantizer to optimize model precision, and runtime components to execute the graph on target hardware. For details refer to the backend documentation and the Export and Lowering tutorial for more information.
│   ├── apple - Apple-specific backends.
├── codegen - Tooling to autogenerate bindings between kernels and the runtime.
├── configurations - Configuration files.
├── devtools - Model profiling, debugging, and inspection. Please refer to the tools documentation for more information.
├── docs - Static docs tooling and documentation source files.
├── examples - Examples of various user flows, such as model export, delegates, and runtime execution.
├── exir - Ahead-of-time library: model capture and lowering APIs. EXport Intermediate Representation (EXIR) is a format for representing the result of torch.export. This directory contains utilities and passes for lowering the EXIR graphs into different dialects and eventually suitable to run on target hardware.
├── extension - Extensions built on top of the runtime.
├── kernels - 1st party kernel implementations.
├── profiler - Utilities for profiling runtime execution.
├── runtime - Core C++ runtime. These components are used to execute the ExecuTorch program. Please refer to the runtime documentation for more information.
├── schema - ExecuTorch PTE file format flatbuffer schemas.
├── scripts - Utility scripts for building libs, size management, dependency management, etc.
├── test - Broad scoped end-to-end tests.
├── third-party - Third-party dependencies.
├── tools - Tools for building ExecuTorch from source, for different built tools (CMake, Buck).
└── util - Various helpers and scripts.
```


## Reference

- https://github.com/pytorch/executorch/blob/main/docs/source/concepts.md
- https://github.com/pytorch/executorch/blob/main/CONTRIBUTING.md

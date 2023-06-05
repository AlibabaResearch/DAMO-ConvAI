from setuptools import Extension, setup
from Cython.Build import cythonize

libraries = [
    "folly",
    "thriftcpp2",
]
extra_libraries = [
    "picard.cpython-37m-x86_64-linux-gnu",
]
extra_library_dirs = [
    "build/lib.linux-x86_64-3.7/picard",
]
extensions = [
    Extension(
        "picard.libpicard",
        [
            "gen-cpp2/Picard.cpp",
            "gen-cpp2/PicardAsyncClient.cpp",
            "gen-cpp2/Picard_processmap_binary.cpp",
            "gen-cpp2/Picard_processmap_compact.cpp",
            "gen-cpp2/picard_constants.cpp",
            "gen-cpp2/picard_data.cpp",
            "gen-cpp2/picard_metadata.cpp",
            "gen-cpp2/picard_types.cpp"
        ],
        include_dirs=[],
        libraries=libraries,
        library_dirs=[],
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.types",
        [
            "picard/types.pyx",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.metadata",
        [
            "picard/metadata.pyx",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.types_fields",
        [
            "picard/types_fields.pyx",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.types_reflection",
        [
            "picard/types_reflection.pyx",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.clients",
        [
            "picard/clients.pyx",
            "picard/clients_wrapper.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.services",
        [
            "picard/services.pyx",
            "picard/services_wrapper.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    ),
    Extension(
        "picard.services_reflection",
        [
            "picard/services_reflection.pyx",
            "/app/third_party/fbthrift/thrift/lib/py3/metadata.cpp",
            "/app/third_party/fbthrift/thrift/lib/py3/enums.cpp",
        ],
        include_dirs=["../."],
        libraries=libraries + extra_libraries,
        library_dirs=extra_library_dirs,
        extra_compile_args = ["--std=c++17"]
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)

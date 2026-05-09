@echo off
set CONFIG=ru/neural_segmenter_32k
if defined PYTHON_EXE (
    set "PYTHON_CMD=%PYTHON_EXE%"
) else (
    set "PYTHON_CMD=python"
)

echo [1/3] Training Tokenizer...
%PYTHON_CMD% scripts/01_train_tokenizer.py --config %CONFIG%
if %ERRORLEVEL% neq 0 (
    echo Tokenizer training failed!
    exit /b %ERRORLEVEL%
)

echo [2/3] Preprocessing Data...
%PYTHON_CMD% scripts/02_preprocess_data.py --config %CONFIG%
if %ERRORLEVEL% neq 0 (
    echo Preprocessing failed!
    exit /b %ERRORLEVEL%
)

echo [3/3] Training Model...
%PYTHON_CMD% scripts/03_train_model.py --config %CONFIG%
if %ERRORLEVEL% neq 0 (
    echo Model training failed!
    exit /b %ERRORLEVEL%
)

echo E2E Pipeline completed successfully!

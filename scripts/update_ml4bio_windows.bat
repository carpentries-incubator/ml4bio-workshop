@REM this assumes that the user has Anaconda installed somewhere on their system, it is on the path
@REM and the ml4bio conda environment has already been installed
CALL activate.bat
CALL activate ml4bio

@REM try updating the ml4bio package
pip install ml4bio --upgrade --upgrade-strategy only-if-needed

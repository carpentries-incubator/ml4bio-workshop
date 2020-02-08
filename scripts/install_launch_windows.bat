@REM this assumes that the user has Anaconda installed somewhere on their system and it is on the path
CALL activate.bat

@REM check whether the user already installed the ml4bio conda environment
@REM by trying to activate the environment
@REM if the environment does not exist conda will return a CondaEnvironmentNotFoundError
CALL activate ml4bio

IF %ERRORLEVEL% NEQ 0 (
  ECHO creating ml4bio environment
  conda env create -f conda_env.yml
  CALL activate ml4bio
  @REM launch the GUI
  @REM this does not run as expected if it is placed outside the IF block
  @REM so it is copied in the IF and ELSE block
  ml4bio
) ELSE (
  @REM launch the GUI
  ml4bio
)

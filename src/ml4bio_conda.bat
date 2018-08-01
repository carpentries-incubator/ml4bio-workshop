@rem this assumes that the user has Anaconda installed somewhere on their system and it is on the path
call activate.bat

@rem this assumes that the user already installed the ml4bio conda environment
@rem we could check whether it is installed and create it if needed
call activate ml4bio

@rem launch the GUI
python ml4bio.py

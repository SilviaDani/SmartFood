
  ## Problems encountered
  
   # MOMENT
   Not usable at the moment. It requires transformers==4.33.3, but Chronos2 needs transformers>=4.41 and the two pakages are incompatible with each other. For now the MOMENT import is inside a try/except ImportError that raise a ValueError with this message: "MOMENT è disabilitato, usa Chronos o TimesFM".

   # TimesFM
   Similar problem that MOMENT have. TimesFM need PyPI that uses JAX as backend wich is incompatible with PyTorch.

   A possible solution for this is to use a different docker container for each model.


  # Modern School Menu Page

  This is a code bundle for Modern School Menu Page. The original project is available at https://www.figma.com/design/OXv2XT6CG9U2w3rmsc4eBO/Modern-School-Menu-Page.

  ## How to run the code locally for testing purpouses (without docker)

  Need 2 terminals *if you do not want to test the training queue*: one running the backend and one running the frontend

  ## Running the frontend

  Run `npm i` to install the dependencies.

  Run `npm run dev` to start the development server.

  ## Running the backend
  (need more testing and research)

  # first activate the conda enviromnent
   `conda activate <name_of_the_env>` (e.g. smartF311)
  # then change directory and go in the backend
   `cd backend`
  # finally run the code
   `python -m smartfood.app`
  
## To test Training Queue

  *Need 2 more terminals*, and you need redis installed locally. Can be found here: https://github.com/microsoftarchive/redis/releases 
  
  Also, remember to update the requirements using:
    `pip install -r requirements.txt`
  Best if installed in an environment

  # Terminal 1: Redis
  `redis-server`

  # Terminal 2: Celery Worker
  `cd backend`
  `celery -A smartfood.celery_app worker --loglevel=info`

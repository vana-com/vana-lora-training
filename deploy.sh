#!/bin/bash

# Check if vana-ai-utils-py directory already exists
if [ -d "vana-ai-utils-py" ]; then
  rm -rf vanautils
fi

# Check if login flag is passed
if [[ "$*" == *--login* ]]; then
  cog login
  echo "Cog is now logged in"
fi

if [[ "$*" == *--prune* ]]; then
  echo "Pruning docker images"
  docker image prune
fi

cp -r ../vana-ai-utils-py/vanautils .

# Push to dev or prod
if [[ "$*" == *--dev* ]]; then
  if [[ "$*" == *--rebuild-base* ]]; then
    python ./scripts/setup.py
  fi
  cog push r8.im/vana-com/vana-lora-training-dev
fi

if [[ "$*" == *--prod* ]]; then
  # Generate 4 random numbers
  random_numbers=$(od -vAn -N4 -tu1 < /dev/urandom | tr -d ' ' | cut -c 1-4)
  
  # Prompt user to repeat back the numbers
  read -p "To deploy to prod, please repeat back the following 4 numbers in order:  $(echo -n $random_numbers) " user_numbers
  
  # Check if user entered the correct numbers
  if [ "$user_numbers" == "$random_numbers" ]; then
    python ./scripts/setup.py
    cog push r8.im/vana-com/vana-lora-training
  else
    echo "Incorrect numbers. Deployment to prod aborted."
    rm -rf vanautils
    exit 1
  fi
fi

# Remove vana-ai-utils-py directory
rm -rf vanautils
FROM pytorch/pytorch

# copy source code into the container
COPY . .

# dependencies
RUN pip install -r requirements.txt

# download datasets into the container once (~400 MB)
RUN python -c "from data import BavarianCrops; BavarianCrops('train'); BavarianCrops('valid'); BavarianCrops('eval')"

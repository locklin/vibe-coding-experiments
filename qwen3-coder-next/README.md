Examples run with my 32 core threadripper with

numactl --interleave=all ./build/bin/llama-server \
  -m ~/qwen3-coder-next-Q4_K_M.gguf \
  --numa distribute \
  --threads 32 \
  -c 8192 \
  --no-mmap \
  --jinja \
  --host 0.0.0.0 --port 8080


and emacs gptel-agent

BNBEM:

  I want to implement this paper as R code in a new repo called BernoulliNBEM. https://www.cs.columbia.edu/~mcollins/em.pdf
  Don't ask me to install playwrite, it's already installed and I don't feel like giving you any passwords. If you can't find it, use something else.
  The idea of the paper is to represent documents by presence or absence of word stems. You can fit the Bernoulli Naive Bayes model to classified documents.
  When new documents are added to the model, we use likelihood and the EM algorithm to increase the accuracy of the model even when there isn't a class label present.
  It should also provide a guess for each new document as to what class it is in, as in the paper.
  You can use any R package which helps. SparseM and tm and other natural language models in the CRAN view on natural language models may be of use. https://cran.r-project.org/web/views/NaturalLanguageProcessing.html
  You may not use any existing R Naive Bayes packages, as all of them are inefficient, and most of them are Gaussian Naive Bayes rather than Bernoulli.
  All of the objects created must be S3, and the package should have the usual test framework using testthat, as well as documentation for the work flow in manual pages and a vignette.
  For now, it doesn't need the full end to end machinery of obtaining new documents and turning them into vectors showing the presence of absence of words; a test data set can use
  artificial Bernoulli distribution vectors correlated with classes. Extra credit if you can do the whole thing though, using websites as text documents (maybe websites pertaining to different kinds of
  sports, with the kind of sport being the class).


FastLogisticLowRankQ:


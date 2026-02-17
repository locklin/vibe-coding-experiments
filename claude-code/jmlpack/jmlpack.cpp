#include "jmlpack.h"

#include <mlpack.hpp>
#include <armadillo>
#include <cstring>
#include <exception>

/*
 * Data layout note:
 * J stores matrices row-major: shape (n_obs, n_feat).
 * Armadillo stores column-major: shape (n_rows, n_cols).
 * J's row-major (n_obs, n_feat) in memory == arma's column-major (n_feat, n_obs).
 * So we construct: arma::mat(ptr, n_feat, n_obs, true)
 * where n_feat = arma rows, n_obs = arma cols (each col is one observation).
 * This is exactly what mlpack expects: features as rows, observations as columns.
 */

/* ========== Linear Regression ========== */

struct LinRegModel {
    mlpack::LinearRegression<>* model;
};

extern "C" void* jmlpack_linreg_create(void) {
    auto* m = new LinRegModel();
    m->model = nullptr;
    return m;
}

extern "C" int jmlpack_linreg_train(void* handle, const double* data, int n_obs,
                                     int n_feat, const double* responses, double lambda) {
    try {
        auto* m = static_cast<LinRegModel*>(handle);
        delete m->model;

        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        arma::rowvec y(responses, (size_t)n_obs);

        m->model = new mlpack::LinearRegression<>(X, y, lambda);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_linreg_predict(void* handle, const double* data, int n_test,
                                       int n_feat, double* predictions_out) {
    try {
        auto* m = static_cast<LinRegModel*>(handle);
        if (!m->model) return 2;

        arma::mat X(data, (size_t)n_feat, (size_t)n_test);
        arma::rowvec preds;
        m->model->Predict(X, preds);

        std::memcpy(predictions_out, preds.memptr(), n_test * sizeof(double));
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_linreg_betas(void* handle, int n_feat, double* betas_out) {
    try {
        auto* m = static_cast<LinRegModel*>(handle);
        if (!m->model) return 2;

        const arma::vec& params = m->model->Parameters();
        /* params(0) = intercept, params(1..n_feat) = feature coefficients */
        int n = (int)params.n_elem;
        if (n > n_feat + 1) n = n_feat + 1;
        for (int i = 0; i < n; i++) betas_out[i] = params(i);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_linreg_destroy(void* handle) {
    auto* m = static_cast<LinRegModel*>(handle);
    if (m) {
        delete m->model;
        delete m;
    }
}

/* ========== Logistic Regression ========== */

struct LogRegModel {
    mlpack::LogisticRegression<>* model;
};

extern "C" void* jmlpack_logreg_create(void) {
    auto* m = new LogRegModel();
    m->model = nullptr;
    return m;
}

extern "C" int jmlpack_logreg_train(void* handle, const double* data, int n_obs,
                                     int n_feat, const int* labels, double lambda) {
    try {
        auto* m = static_cast<LogRegModel*>(handle);
        delete m->model;

        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        arma::Row<size_t> y((size_t)n_obs);
        for (int i = 0; i < n_obs; i++) y(i) = (size_t)labels[i];

        m->model = new mlpack::LogisticRegression<>(X, y, lambda);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_logreg_classify(void* handle, const double* data, int n_test,
                                        int n_feat, int* predictions_out) {
    try {
        auto* m = static_cast<LogRegModel*>(handle);
        if (!m->model) return 2;

        arma::mat X(data, (size_t)n_feat, (size_t)n_test);
        arma::Row<size_t> preds;
        m->model->Classify(X, preds);

        for (int i = 0; i < n_test; i++) predictions_out[i] = (int)preds(i);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_logreg_betas(void* handle, int n_feat, double* betas_out) {
    try {
        auto* m = static_cast<LogRegModel*>(handle);
        if (!m->model) return 2;

        const arma::rowvec& params = m->model->Parameters();
        /* params(0) = intercept/bias, params(1..n_feat) = feature coefficients */
        int n = (int)params.n_elem;
        if (n > n_feat + 1) n = n_feat + 1;
        for (int i = 0; i < n; i++) betas_out[i] = params(i);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_logreg_destroy(void* handle) {
    auto* m = static_cast<LogRegModel*>(handle);
    if (m) {
        delete m->model;
        delete m;
    }
}

/* ========== Decision Tree ========== */

struct DTreeModel {
    mlpack::DecisionTree<>* model;
};

extern "C" void* jmlpack_dtree_create(void) {
    auto* m = new DTreeModel();
    m->model = nullptr;
    return m;
}

extern "C" int jmlpack_dtree_train(void* handle, const double* data, int n_obs,
                                    int n_feat, const int* labels, int n_classes,
                                    int min_leaf, int max_depth) {
    try {
        auto* m = static_cast<DTreeModel*>(handle);
        delete m->model;

        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        arma::Row<size_t> y((size_t)n_obs);
        for (int i = 0; i < n_obs; i++) y(i) = (size_t)labels[i];

        m->model = new mlpack::DecisionTree<>(X, y, (size_t)n_classes,
                                               (size_t)min_leaf, 1e-7,
                                               (size_t)max_depth);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_dtree_classify(void* handle, const double* data, int n_test,
                                       int n_feat, int* predictions_out) {
    try {
        auto* m = static_cast<DTreeModel*>(handle);
        if (!m->model) return 2;

        arma::mat X(data, (size_t)n_feat, (size_t)n_test);
        arma::Row<size_t> preds;
        m->model->Classify(X, preds);

        for (int i = 0; i < n_test; i++) predictions_out[i] = (int)preds(i);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_dtree_destroy(void* handle) {
    auto* m = static_cast<DTreeModel*>(handle);
    if (m) {
        delete m->model;
        delete m;
    }
}

/* ========== Random Forest ========== */

struct RForestModel {
    mlpack::RandomForest<>* model;
};

extern "C" void* jmlpack_rforest_create(void) {
    auto* m = new RForestModel();
    m->model = nullptr;
    return m;
}

extern "C" int jmlpack_rforest_train(void* handle, const double* data, int n_obs,
                                      int n_feat, const int* labels, int n_classes,
                                      int n_trees, int min_leaf, int max_depth) {
    try {
        auto* m = static_cast<RForestModel*>(handle);
        delete m->model;

        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        arma::Row<size_t> y((size_t)n_obs);
        for (int i = 0; i < n_obs; i++) y(i) = (size_t)labels[i];

        m->model = new mlpack::RandomForest<>(X, y, (size_t)n_classes,
                                               (size_t)n_trees, (size_t)min_leaf,
                                               1e-7, (size_t)max_depth);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_rforest_classify(void* handle, const double* data, int n_test,
                                         int n_feat, int* predictions_out) {
    try {
        auto* m = static_cast<RForestModel*>(handle);
        if (!m->model) return 2;

        arma::mat X(data, (size_t)n_feat, (size_t)n_test);
        arma::Row<size_t> preds;
        m->model->Classify(X, preds);

        for (int i = 0; i < n_test; i++) predictions_out[i] = (int)preds(i);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_rforest_destroy(void* handle) {
    auto* m = static_cast<RForestModel*>(handle);
    if (m) {
        delete m->model;
        delete m;
    }
}

/* ========== K-Means ========== */

struct KMeansModel {
    int dummy;
};

extern "C" void* jmlpack_kmeans_create(void) {
    return new KMeansModel();
}

extern "C" int jmlpack_kmeans_cluster(void* handle, const double* data, int n_obs,
                                       int n_feat, int k, int* assignments_out,
                                       double* centroids_out) {
    try {
        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        arma::Row<size_t> assignments;
        arma::mat centroids;
        mlpack::KMeans<> km;
        km.Cluster(X, (size_t)k, assignments, centroids);

        for (int i = 0; i < n_obs; i++) assignments_out[i] = (int)assignments(i);

        /* centroids: (n_feat, k) column-major from arma
           Output: row-major (k, n_feat) for J */
        for (int c = 0; c < k; c++) {
            for (int f = 0; f < n_feat; f++) {
                centroids_out[c * n_feat + f] = centroids(f, c);
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_kmeans_destroy(void* handle) {
    delete static_cast<KMeansModel*>(handle);
}

/* ========== KNN ========== */

struct KNNModel {
    arma::mat refData;
    bool trained;
};

extern "C" void* jmlpack_knn_create(void) {
    auto* m = new KNNModel();
    m->trained = false;
    return m;
}

extern "C" int jmlpack_knn_train(void* handle, const double* data, int n_obs, int n_feat) {
    try {
        auto* m = static_cast<KNNModel*>(handle);
        m->refData = arma::mat(data, (size_t)n_feat, (size_t)n_obs);
        m->trained = true;
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int jmlpack_knn_search(void* handle, const double* query, int n_query,
                                   int n_feat, int k, int* neighbors_out,
                                   double* distances_out) {
    try {
        auto* m = static_cast<KNNModel*>(handle);
        if (!m->trained) return 2;

        arma::mat Q(query, (size_t)n_feat, (size_t)n_query);
        arma::Mat<size_t> neighbors;
        arma::mat distances;

        mlpack::KNN knn(m->refData);
        knn.Search(Q, (size_t)k, neighbors, distances);

        /* neighbors and distances: (k, n_query) column-major
           Output: row-major (n_query, k) for J */
        for (int i = 0; i < n_query; i++) {
            for (int j = 0; j < k; j++) {
                neighbors_out[i * k + j] = (int)neighbors(j, i);
                distances_out[i * k + j] = distances(j, i);
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_knn_destroy(void* handle) {
    delete static_cast<KNNModel*>(handle);
}

/* ========== PCA ========== */

struct PCAModel {
    bool scaleData;
};

extern "C" void* jmlpack_pca_create(int scale_data) {
    auto* m = new PCAModel();
    m->scaleData = (scale_data != 0);
    return m;
}

extern "C" int jmlpack_pca_apply(void* handle, const double* data, int n_obs,
                                  int n_feat, double* output, int new_dim,
                                  double* eigvals_out, double* varret_out) {
    try {
        auto* m = static_cast<PCAModel*>(handle);

        /* First do full PCA to get eigenvalues */
        arma::mat X(data, (size_t)n_feat, (size_t)n_obs);
        mlpack::PCA<> pca(m->scaleData);

        arma::mat transformedData;
        arma::vec eigvals;
        arma::mat eigvecs;
        pca.Apply(X, transformedData, eigvals, eigvecs);

        /* transformedData: (n_feat, n_obs) column-major, sorted by eigenvalue desc.
           We want the first new_dim components.
           Output to J as row-major (n_obs, new_dim). */
        for (int i = 0; i < n_obs; i++) {
            for (int j = 0; j < new_dim; j++) {
                output[i * new_dim + j] = transformedData(j, i);
            }
        }

        /* Copy eigenvalues (first new_dim, already sorted desc) */
        int n_eig = (int)eigvals.n_elem;
        if (n_eig > new_dim) n_eig = new_dim;
        for (int i = 0; i < n_eig; i++) eigvals_out[i] = eigvals(i);

        /* Compute variance retained */
        double total_var = arma::accu(eigvals);
        double retained_var = 0.0;
        for (int i = 0; i < n_eig; i++) retained_var += eigvals(i);
        *varret_out = (total_var > 0) ? retained_var / total_var : 0.0;

        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void jmlpack_pca_destroy(void* handle) {
    delete static_cast<PCAModel*>(handle);
}

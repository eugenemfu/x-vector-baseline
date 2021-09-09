#include "clustering.h"

FamilyClustering::FamilyClustering(std::string initDataPath, std::vector<EnrollTemplatePtr> array,
                                   float minVerifyScore) {
    sessionManager = voicesdk::iot::SessionManager::Create(initDataPath);
    matcher = sessionManager->CreateVoiceTemplateMatcher();
    templateArray = array;
    templateNumber = array.size();
    unlockScore = minVerifyScore;
}

void FamilyClustering::calculateClusterTemplates() {
    for (int i = 1; i <= clustersNumber; ++i) {
        auto session = sessionManager->CreateEnrollTemplateSession();
        for (size_t j = 0; j < templateNumber; ++j) {
            if (label[j] == i) {
                session->AddEntry(templateArray[j]);
            }
        }
        clusterTemplates.push_back(session->Complete());
    }
}

int FamilyClustering::verify(const VerifyTemplatePtr &verifyTemplate) {
    bool unlocked = false; // false;
    float max_score = 0;
    int best_candidate = 0;
    for (size_t i = 0; i < clustersNumber; ++i) {
        auto result = matcher->Match(clusterTemplates[i], verifyTemplate);
        if (result.verification_score > unlockScore) {
            unlocked = true;
        }
        if (result.verification_score > max_score) {
            max_score = result.verification_score;
            best_candidate = i;
        }
    }
    return unlocked ? best_candidate + 1 : 0;
}

FamilyClustering::VerifyTemplatePtr FamilyClustering::enrollToVerifyTemplate(const EnrollTemplatePtr &enrollTemplate) {
    std::stringstream str;
    enrollTemplate->Serialize(str);
    str.seekg(0, std::ios::beg);
    return voicesdk::iot::VerifyVoiceTemplate::Deserialize(str);
}


FamilyDBSCAN::FamilyDBSCAN(std::string initDataPath, std::vector<EnrollTemplatePtr> array, float minVerifyScore,
                           int minPts, float minScore) :
        FamilyClustering(initDataPath, array, minVerifyScore),
        neighborScore(minScore) {
    clustersNumber = 0;

    label = new int[templateNumber];
    for (size_t i = 0; i < templateNumber; ++i) {
        label[i] = -1;
    }

    for (size_t i = 0; i < templateNumber; ++i) {
        if (label[i] != -1) {
            continue;
        }
        std::cout << std::endl << i << "/" << templateNumber << " ";

        std::vector<size_t> neighbors = findNeighbors(i);
        if (neighbors.size() < minPts) {
            label[i] = 0;
            continue;
        }
        label[i] = ++clustersNumber;
        for (size_t j = 0; j < neighbors.size(); ++j) {
            //std::cout << j << "/" << neighbors.size() << std::endl;
            if (label[neighbors[j]] == 0) {
                label[neighbors[j]] = clustersNumber;
            }
            if (label[neighbors[j]] != -1) {
                continue;
            }
            label[neighbors[j]] = clustersNumber;
            std::vector<size_t> neighborsNeighbors = findNeighbors(neighbors[j]);
            if (neighborsNeighbors.size() >= minPts) {
                for (size_t &neighborsNeighbor: neighborsNeighbors) {
                    bool repeat = false;
                    for (size_t neighbor: neighbors) {
                        if (neighborsNeighbor == neighbor) {
                            repeat = true;
                            break;
                        }
                    }
                    if (!repeat) {
                        neighbors.push_back(neighborsNeighbor);
                    }
                }
            }
        }
    }

    // std::cout << std::endl;

    // Save centers of clusters
    calculateClusterTemplates();
}

std::vector<size_t> FamilyDBSCAN::findNeighbors(size_t i) {
    if (cachedNeighbors.count(i)) {
        return cachedNeighbors.at(i);
    }
    std::vector<size_t> neighbors;
    VerifyTemplatePtr template_i = enrollToVerifyTemplate(templateArray[i]);
    for (size_t j = 0; j < templateNumber; j++) {
//            if (i == j) {
//                continue;
//            }
        auto result = matcher->Match(templateArray[j], template_i);
        if (result.verification_score > neighborScore) {
            neighbors.push_back(j);
        }
    }
    cachedNeighbors.insert({i, neighbors});
    //std::cout << neighbors.size() << " ";
    return neighbors;
}


FamilyAgglClust::FamilyAgglClust(std::string initDataPath, std::vector<EnrollTemplatePtr> array, float minVerifyScore,
                                 int minClusterSize, float minMergeScore) :
        FamilyClustering(initDataPath, array, minVerifyScore),
        minClusterSize(minClusterSize) {
    label = new int[templateNumber];
    for (int i = 0; i < templateNumber; ++i) {
        label[i] = i + 1;
    }

    scoreMatrix = new float *[templateNumber];
    for (size_t i = 0; i < templateNumber; ++i) {
        scoreMatrix[i] = new float[templateNumber];
        for (size_t j = 0; j < templateNumber; ++j) {
            if (i == j) {
                scoreMatrix[i][j] = 0;
            } else if (j < i) {
                scoreMatrix[i][j] = scoreMatrix[j][i];
            } else {
                auto result = matcher->Match(templateArray[i], enrollToVerifyTemplate(templateArray[j]));
                scoreMatrix[i][j] = result.verification_score;
            }
        }
    }

    clustersNumber = templateNumber;
    //std::cout << "Clusters number: " << clustersNumber << std::endl;

    while (clustersNumber > 1) {
        float maxScoreBetweenClusters = -std::numeric_limits<float>::infinity();
        int i_max = 0;
        int j_max = 0;
        for (int i = 1; i <= clustersNumber; ++i) {
            for (int j = i + 1; j <= clustersNumber; ++j) {
                //std::cout << "Cluster pair " << i << " - " << j << " processing..." << std::endl;
                float score = averageScoreBetweenClusters(i, j);
                if (score > maxScoreBetweenClusters) {
                    maxScoreBetweenClusters = score;
                    i_max = i;
                    j_max = j;
                }
            }
        }
        if (maxScoreBetweenClusters < minMergeScore) {
            break;
        }
        mergeClusters(i_max, j_max);
        //std::cout << "Clusters number: " << clustersNumber << std::endl;
    }

    // std::cout << std::endl;

    removeSmallClusters();

    calculateClusterTemplates();
}

FamilyAgglClust::~FamilyAgglClust() {
    delete[] label;
    for (size_t i = 0; i < templateNumber; ++i) {
        delete[] scoreMatrix[i];
    }
    delete[] scoreMatrix;
}

float FamilyAgglClust::averageScoreBetweenClusters(int i, int j) {
    std::vector<size_t> points_i;
    std::vector<size_t> points_j;
    for (size_t k = 0; k < templateNumber; ++k) {
        if (label[k] == i) {
            points_i.push_back(k);
        }
        if (label[k] == j) {
            points_j.push_back(k);
        }
    }

    float sum = 0;
    int count = 0;
    for (size_t k: points_i) {
        for (size_t l: points_j) {
            sum += scoreMatrix[k][l];
            ++count;
        }
    }
    return sum / count;
}

void FamilyAgglClust::mergeClusters(int i, int j) {  // i < j
    for (size_t k = 0; k < templateNumber; ++k) {
        if (label[k] == j) {
            label[k] = i;
        } else if (label[k] > j) {
            --label[k];
        }
    }
    --clustersNumber;
}

void FamilyAgglClust::removeSmallClusters() {
    for (int i = 1; i <= clustersNumber; ++i) {
        int count = 0;
        for (size_t k = 0; k < templateNumber; ++k) {
            if (label[k] == i) {
                ++count;
            }
        }
        if (count < minClusterSize) {
            for (size_t k = 0; k < templateNumber; ++k) {
                if (label[k] == i) {
                    label[k] = 0;
                } else if (label[k] > i) {
                    --label[k];
                }
            }
            --clustersNumber;
            --i;
        }
    }
}
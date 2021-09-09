#ifndef DEVICEUNLOCKSDK_EXAMPLES_CLUSTERING_H
#define DEVICEUNLOCKSDK_EXAMPLES_CLUSTERING_H

#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <map>
#include <algorithm>

#include <voicesdk/iot/iot.h>

class FamilyClustering {
public:
    using EnrollTemplatePtr = std::shared_ptr<voicesdk::iot::EnrollVoiceTemplate>;
    using VerifyTemplatePtr = std::shared_ptr<voicesdk::iot::VerifyVoiceTemplate>;
    using SessionManagerPtr = std::shared_ptr<voicesdk::iot::SessionManager>;
    using MatcherPtr = std::shared_ptr<voicesdk::iot::VoiceTemplateMatcher>;

    std::vector<EnrollTemplatePtr> templateArray;
    size_t templateNumber;
    int clustersNumber;
    int* label;
    SessionManagerPtr sessionManager;
    MatcherPtr matcher;
    std::vector<EnrollTemplatePtr> clusterTemplates;
    float unlockScore;

    FamilyClustering(std::string initDataPath, std::vector<EnrollTemplatePtr> array, float minVerifyScore);
    ~FamilyClustering() = default;

    void calculateClusterTemplates();
    int verify(const VerifyTemplatePtr& verifyTemplate);
    static VerifyTemplatePtr enrollToVerifyTemplate(const EnrollTemplatePtr& enrollTemplate);

};


class FamilyDBSCAN: public FamilyClustering {
public:
    std::map<size_t, std::vector<size_t>> cachedNeighbors;
    float neighborScore;

    FamilyDBSCAN(std::string initDataPath, std::vector<EnrollTemplatePtr> array, float minVerifyScore, int minPts, float minScore);
    ~FamilyDBSCAN() = default;

    std::vector<size_t> findNeighbors(size_t i);
};


class FamilyAgglClust: public FamilyClustering {
public:
    int minClusterSize;
    float** scoreMatrix;

    FamilyAgglClust(std::string initDataPath, std::vector<EnrollTemplatePtr> array, float minVerifyScore, int minClusterSize, float minMergeScore);
    ~FamilyAgglClust();

    float averageScoreBetweenClusters(int i, int j);
    void mergeClusters(int i, int j);
    void removeSmallClusters();
};

#endif //DEVICEUNLOCKSDK_EXAMPLES_CLUSTERING_H

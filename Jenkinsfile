pipeline {
    agent {
        node {
            label 'Agent02'
        }
    }
    options {
        timestamps()
        timeout(time: 5, unit: 'MINUTES')
    }

    environment {
        GRAALVM_21_JAVA_HOME="/var/lib/jenkins/workspace/TornadoVM/dependencies/jdks/graal-23.1.0/graalvm-community-openjdk-21.0.1+12.1"
        TORNADO_ROOT="/var/lib/jenkins/workspace/TornadoVM/TornadoVM-pipeline"
        PATH="/opt/maven/bin:/var/lib/jenkins/workspace/TornadoVM/TornadoVM-pipeline/bin/bin:$PATH"
        TORNADO_SDK="/var/lib/jenkins/workspace/TornadoVM/TornadoVM-pipeline/bin/sdk"
        CMAKE_ROOT="/var/lib/jenkins/workspace/TornadoVM/dependencies/cmake/cmake-3.25.2-linux-x86_64"
        TORNADO_DOCKER_SCRIPT="/var/lib/jenkins/workspace/TornadoVM/dependencies/docker-tornadovm"
        DOCKER_IMAGE_REGISTRY="beehivelab/tornadovm-polyglot-graalpy-23.1.0-oneapi-intel-container"
        DOCKER_IMAGE_TAG="tango"
    }
    stages {
        stage('Pull the Dynamic-Intelligent-Execution container') {
            sh 'docker pull "$DOCKER_IMAGE_REGISTRY:$DOCKER_IMAGE_TAG"'
        }
        stage('Prepare build and test') {
            steps {
                script {
                    runJDK21()
                }
            }
        }
        stage('Docker Remove Image locally') {
        steps {
                sh 'docker rmi "$DOCKER_IMAGE_REGISTRY:$DOCKER_IMAGE_TAG"'
            }
        }
    }
}

void runJDK21() {
    stage('OpenJDK 21') {
        withEnv(["JAVA_HOME=${JDK_21_JAVA_HOME}"]) {
            buildAndTest("GraalVM JDK 21", "graal-jdk-21")
        }
    }
}

void buildAndTest(String JDK, String tornadoProfile) {
    echo "-------------------------"
    echo "JDK used " + JDK
    echo "Tornado profile " + tornadoProfile
    echo "-------------------------"
    stage('Build with ' + JDK) {
        sh "make ${tornadoProfile} BACKEND=ptx,opencl,spirv"
    }
    stage('TornadoVM Tests') {
        timeout(time: 12, unit: 'MINUTES') {
            sh 'tornado --devices'
            sh '${TORNADO_DOCKER_SCRIPT}/dynamic-intelligent-execution-intel.sh --test'
        }
    }
    stage("TornadoVM Integration Tests (Java-Python)") {
        timeout(time: 12, unit: 'MINUTES') {
            sh 'cd ${TORNADO_DOCKER_SCRIPT} && ${TORNADO_DOCKER_SCRIPT}/dynamic-intelligent-execution-intel.sh --test_integration'
        }
    }
}

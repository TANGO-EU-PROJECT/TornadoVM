pipeline {
    agent any
    options {
        timestamps()
        timeout(time: 5, unit: 'MINUTES')
    }

    environment {
        GRAALVM_21_JAVA_HOME="/opt/jenkins/jdks/graal-23.1.0/graalvm-community-openjdk-21.0.1+12.1"
        TORNADO_ROOT="/var/lib/jenkins/workspace/TornadoVM-pipeline"
        PATH="/opt/maven/bin:/var/lib/jenkins/workspace/TornadoVM-pipeline/bin/bin:$PATH"
        TORNADO_SDK="/var/lib/jenkins/workspace/TornadoVM-pipeline/bin/sdk"
        CMAKE_ROOT="/opt/jenkins/cmake-3.25.2-linux-x86_64"
        TORNADO_DOCKER_SCRIPT="/var/lib/jenkins/docker-tornado"
    }
    stages {
        stage('Pull the Dynamic-Intelligent-Execution container') {
            sh "docker pull beehivelab/tornadovm-polyglot-graalpy-23.1.0-oneapi-intel-container:tango"
        }
        stage('Prepare build and test') {
            steps {
                script {
                    runJDK21()
                }
            }
        }
    }
}

void runJDK21() {
    stage('OpenJDK 21') {
        withEnv(["JAVA_HOME=${JDK_21_JAVA_HOME}"]) {
            buildAndTest("OpenJDK 21", "jdk21")
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
            sh '${TORNADO_DOCKER_SCRIPT}/dynamic-intelligent-execution-intel.sh tornado-test -V uk.ac.manchester.tornado.unittests.profiler.TestProfiler'
        }
    }
    stage("TornadoVM Integration Tests (Java-Python)") {
        timeout(time: 12, unit: 'MINUTES') {
            sh 'cd ${TORNADO_DOCKER_SCRIPT} && ${TORNADO_DOCKER_SCRIPT}/dynamic-intelligent-execution-intel.sh tornado --truffle python example/polyglot-examples/kmeans.py'
        }
    }
}

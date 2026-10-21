// CI/CD for whitebox-medmnist, on a Jenkins running on the machine with the GPU.
pipeline {
    agent any

    parameters {
        booleanParam(name: 'RETRAIN', defaultValue: false, description: 'retrain the 128 px models and rerun the sanity checks (GPU, ~30 min)')
    }

    environment {
        PATH = "${env.HOME}/.local/bin:${env.HOME}/.local/opt/node/bin:${env.PATH}"
    }

    stages {
        stage('Install') {
            steps { sh 'uv sync --group mlops' }
        }

        stage('Lint and unit tests') {
            steps {
                sh 'uv run ruff check src scripts tests'
                sh 'mkdir -p reports && uv run --group mlops pytest -q --junitxml=reports/pytest.xml'
            }
            post {
                always { junit 'reports/pytest.xml' }
            }
        }

        stage('Web build') {
            steps {
                dir('web') { sh 'npm ci && npx ng build' }
            }
        }

        stage('Retrain and sanity-check (GPU)') {
            when { expression { params.RETRAIN } }
            steps {
                sh '''
                    for ds in pneumoniamnist bloodmnist; do
                        uv run scripts/train.py $ds --size 128
                        uv run scripts/sanity.py $ds --size 128
                    done
                '''
            }
        }

        stage('Quality gate') {
            steps { sh 'uv run --group mlops scripts/quality_gate.py --size 128' }
        }
    }
}

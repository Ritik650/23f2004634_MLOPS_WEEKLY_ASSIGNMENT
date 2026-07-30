-- wrk load-test script: POST /predict with a JSON body
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"feature1": 1.0, "feature2": 2.0}'

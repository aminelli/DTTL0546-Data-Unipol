@echo off


docker run -d --name=grafana-comm -h grafana-comm -p 3001:3000 grafana/grafana

docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise



package main

import (
	"flag"
	"log"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/server"
)

func main() {
	var port string

	flag.StringVar(&port, "port", ":8080", "The port which the master service binds to.")
	router := server.NewRouter()

	// Listen and serve on defined port
	log.Printf("Listening on port %s", port)
	router.Run(":" + port)
}

package flagset

import (
	"flag"
	"log/slog"
	"os"
)

type Flag struct {
	_              struct{}
	addrFlag       string
	debugFlag      bool
	jsonFlag       bool
	milvusAddr     string
	modelGardenURL string
	modelName      string
	versionFlag    bool
}

func (f *Flag) Addr() string {
	return f.addrFlag
}

func (f *Flag) Debug() bool {
	return f.debugFlag
}

func (f *Flag) JSON() bool {
	return f.jsonFlag
}

func (f *Flag) MilvusAddr() string {
	return f.milvusAddr
}

func (f *Flag) ModelGardenURL() string {
	return f.modelGardenURL
}

func (f *Flag) ModelName() string {
	return f.modelName
}

func (f *Flag) Version() bool {
	return f.versionFlag
}

func ParseFlag(args []string) (*Flag, error) {
	f := &Flag{}
	fs := flag.NewFlagSet(args[0], flag.ExitOnError)
	fs.StringVar(&f.addrFlag, "addr", ":1428", "mcp server address")
	fs.BoolVar(&f.debugFlag, "debug", false, "enable debug logging")
	fs.BoolVar(&f.jsonFlag, "json", false, "enable JSON logging")
	fs.StringVar(&f.milvusAddr, "milvus-addr", "localhost:19530", "milvus vector database address")
	fs.StringVar(&f.modelGardenURL, "model-garden-url", "https://modelgarden.com/embeddings", "Model Garden embeddings endpoint URL")
	fs.StringVar(&f.modelName, "model-name", "embeddinggemma-300m", "Model Garden model name for embeddings")
	fs.BoolVar(&f.versionFlag, "version", false, "print version and exit")

	err := fs.Parse(args[1:])
	if err != nil {
		return nil, err
	}
	f.initializeLogger()

	return f, nil
}

func (f *Flag) initializeLogger() {
	var opts slog.HandlerOptions
	if f.Debug() {
		opts.Level = slog.LevelDebug
	} else {
		opts.Level = slog.LevelInfo
	}
	var handler slog.Handler
	if f.JSON() {
		handler = slog.NewJSONHandler(os.Stdout, &opts)
	} else {
		handler = slog.NewTextHandler(os.Stdout, &opts)
	}
	slog.SetDefault(slog.New(handler))
}

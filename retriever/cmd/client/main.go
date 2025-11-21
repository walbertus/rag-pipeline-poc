package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/gopaytech/rag-pipeline-poc/retriever/internal/flagset"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	BuildTime = "unknown"
	Name      = "retriever-mcp-server"
	Version   = "dev"
)

func initializeLogger(debug, json bool) {
	var opts slog.HandlerOptions
	if debug {
		opts.Level = slog.LevelDebug
	} else {
		opts.Level = slog.LevelInfo
	}
	var handler slog.Handler
	if json {
		handler = slog.NewJSONHandler(os.Stdout, &opts)
	} else {
		handler = slog.NewTextHandler(os.Stdout, &opts)
	}
	slog.SetDefault(slog.New(handler))
}

func run(ctx context.Context, args []string) error {
	flag, err := flagset.ParseFlag(args[:])
	if err != nil {
		return err
	}

	initializeLogger(flag.Debug(), flag.JSON())

	if flag.Version() {
		slog.LogAttrs(ctx,
			slog.LevelInfo,
			"version info",
			slog.String("name", Name),
			slog.String("version", Version),
			slog.String("build_time", BuildTime),
		)
		return nil
	}

	slog.LogAttrs(ctx,
		slog.LevelInfo,
		"starting server",
		slog.String("name", Name),
		slog.String("version", Version),
		slog.String("build_time", BuildTime),
		slog.String("addr", flag.Addr()),
	)

	client := mcp.NewClient(&mcp.Implementation{
		Name:    Name,
		Version: Version,
	}, &mcp.ClientOptions{
		LoggingMessageHandler: func(ctx context.Context, lmr *mcp.LoggingMessageRequest) {
			slog.LogAttrs(ctx,
				slog.LevelInfo,
				"logging message",
				slog.String("session", lmr.GetSession().ID()),
			)
		},
	})

	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: "http://localhost" + flag.Addr()}, &mcp.ClientSessionOptions{})
	if err != nil {
		return err
	}
	defer session.Close()
	result, err := session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return err
	}
	for _, tool := range result.Tools {
		slog.LogAttrs(ctx,
			slog.LevelInfo,
			"available tool",
			slog.Group("tool",
				slog.String("name", tool.Name),
				slog.String("description", tool.Description),
				slog.Any("input", tool.InputSchema),
				slog.Any("output", tool.OutputSchema),
			),
		)
	}

	result2, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name: "query",
		Arguments: map[string]any{
			"query": "example",
		},
	})
	if err != nil {
		return err
	}

	slog.LogAttrs(ctx,
		slog.LevelInfo,
		"call tool result",
		slog.Any("result", result2),
	)
	return nil
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	if err := run(ctx, os.Args[:]); err != nil {
		slog.Default().LogAttrs(ctx,
			slog.LevelError,
			"run failed",
			slog.String("error", err.Error()),
		)
		os.Exit(1)
	}
}

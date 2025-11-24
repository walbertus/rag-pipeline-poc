package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gopaytech/rag-pipeline-poc/retriever/internal/flagset"
	"github.com/gopaytech/rag-pipeline-poc/retriever/internal/model_garden"
	"github.com/gopaytech/rag-pipeline-poc/retriever/internal/tools"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/grpc"
)

var (
	BuildTime = "unknown"
	Name      = "retriever-mcp-server"
	Version   = "dev"
)

func run(ctx context.Context, args []string) error {
	flag, err := flagset.ParseFlag(args[:])
	if err != nil {
		return err
	}

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

	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: flag.MilvusAddr(),
		DialOptions: []grpc.DialOption{
			grpc.WithUnaryInterceptor(func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
				n := time.Now()
				err := invoker(ctx, method, req, reply, cc, opts...)
				slog.Default().WithGroup("milvus_client").LogAttrs(ctx,
					slog.LevelDebug,
					"milvus grpc request",
					slog.String("method", method),
					slog.Any("req", req),
					slog.Any("reply", reply),
					slog.Any("error", err),
					slog.Duration("duration", time.Since(n)),
				)
				return err
			}),
		},
	})
	if err != nil {
		return err
	}
	defer client.Close(ctx)

	server := mcp.NewServer(&mcp.Implementation{
		Name:    Name,
		Version: Version,
	}, &mcp.ServerOptions{
		Logger: slog.Default().WithGroup("mcp_server"),
	})

	tools.RegisterTools(slog.Default(), server, client,
		model_garden.NewModelGarden(
			&http.Client{},
			slog.Default(),
			flag.ModelGardenURL(),
			flag.ModelName(),
		),
	)

	errCh := make(chan error, 1)
	go func() {
		if err := http.ListenAndServe(flag.Addr(),
			mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
				return server
			}, &mcp.StreamableHTTPOptions{
				Logger: slog.Default().WithGroup("mcp_http"),
			}),
		); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.LogAttrs(ctx,
				slog.LevelError,
				"failed to start http server",
				slog.String("error", err.Error()),
			)
			errCh <- err
		}
	}()

	select {
	case <-ctx.Done():
		slog.LogAttrs(ctx,
			slog.LevelInfo,
			"shutting down server",
		)
		return nil
	case err := <-errCh:
		return err
	}
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

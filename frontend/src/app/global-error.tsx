'use client'

export default function GlobalError({
    error,
    reset,
}: {
    error: Error & { digest?: string }
    reset: () => void
}) {
    console.error('Global app error:', error)

    return (
        <html lang="en">
            <body>
                <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 px-6">
                    <div className="max-w-md w-full rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                        <h2 className="text-lg font-semibold text-slate-900">应用发生错误</h2>
                        <p className="mt-2 text-sm text-slate-600">
                            组件渲染出现异常，请重试。
                        </p>
                        <button
                            onClick={reset}
                            className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700"
                        >
                            重新加载
                        </button>
                    </div>
                </div>
            </body>
        </html>
    )
}

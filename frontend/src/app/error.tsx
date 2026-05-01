'use client'

type ErrorProps = {
    error: Error & { digest?: string }
    reset: () => void
}

export default function Error({ error, reset }: ErrorProps) {
    console.error('Route error boundary:', error)

    return (
        <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 px-6">
            <div className="max-w-md w-full rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900">页面发生错误</h2>
                <p className="mt-2 text-sm text-slate-600">
                    请稍后重试。如果问题持续，请刷新页面。
                </p>
                <button
                    onClick={reset}
                    className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700"
                >
                    重试
                </button>
            </div>
        </div>
    )
}

import { Tagged, Untagged } from "prelude";
import { match } from "ts-pattern";

export interface Data<A> extends Tagged<"Data"> {
  data: A;
}
export const Data = <A>(spec: Untagged<Data<A>>): Data<A> => ({
  ...spec,
  _tag: "Data",
});

export interface Error<A> extends Tagged<"Error"> {
  error: A;
}
export const Error = <A>(spec: Untagged<Error<A>>): Error<A> => ({
  ...spec,
  _tag: "Error",
});

export interface Init extends Tagged<"Init"> {}
export const Init: Init = { _tag: "Init" };

export interface Loading extends Tagged<"Loading"> {}
export const Loading: Loading = { _tag: "Loading" };

export type RemoteData<E, A> = Data<A> | Error<E> | Loading | Init;

export const fold =
  <E, A, R>(fn: {
    onData: (data: A) => R;
    onError: (error: E) => R;
    onLoading: () => R;
    onInit: () => R;
  }) =>
  (rd: RemoteData<E, A>): R =>
    match(rd)
      .with({ _tag: "Data" }, ({ data }) => fn.onData(data))
      .with({ _tag: "Error" }, ({ error }) => fn.onError(error))
      .with({ _tag: "Init" }, () => fn.onInit())
      .with({ _tag: "Loading" }, () => fn.onLoading())
      .exhaustive();

export const map =
  <E, A, B>(fn: (x: A) => B) =>
  (rd: RemoteData<E, A>): RemoteData<E, B> =>
    rd._tag === "Data" ? Data({ data: fn(rd.data) }) : rd;
